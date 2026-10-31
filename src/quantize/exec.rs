//! Author: Joe Ricks
//! File: src/quantize/exec.rs
//! Description: Quantizes a GGUF model by calling llama.cpp's quantizer.

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Once;

use llama_cpp_sys_2 as sys;

use super::{cache_refusal, in_hub_cache, prepare, Ftype, QuantError, QuantReport, QuantSpec};
use crate::load::Model;

static BACKEND_INIT: Once = Once::new();

/// Purpose: This function writes a quantized copy of a GGUF model using llama.cpp's
/// quantizer.
///
/// Inputs:
/// - `input`: the model file, or the first part of a split model.
/// - `output`: the file to write, or for a split output the name its parts are made from.
/// - `spec`: the file type to make and the options.
///
/// Outputs: a [`QuantReport`] of what was written, or a [`QuantError`].
pub fn quantize(input: &Path, output: &Path, spec: &QuantSpec) -> Result<QuantReport, QuantError> {
    let model = Model::open(input).map_err(|e| QuantError::BadInput(format!("cannot open the input: {e}")))?;
    let job = prepare(&model, output, spec)?;
    check_outputs_on_disk(&model, &job.outputs)?;

    let inp = c_path(&model.paths[0])?;
    let out = c_path(&job.target)?;
    let nthread = i32::try_from(spec.nthread).unwrap_or(i32::MAX);

    // SAFETY: `inp` and `out` are NUL-terminated C strings that outlive the call, and
    // `params` is the struct llama.cpp's own default constructor returned, with only
    // its plain-value fields changed; its four pointer fields (imatrix, kv_overrides,
    // tt_overrides, prune_layers) stay null, which llama.cpp checks before reading.
    // `llama_model_quantize` catches every C++ exception and returns 1 instead, so no
    // unwind crosses the FFI boundary. `llama_backend_init` runs once, before the first
    // quantize, as `llama-quantize` runs it; the closure is inside this block, so it is
    // covered by the same argument.
    let status = unsafe {
        BACKEND_INIT.call_once(|| sys::llama_backend_init());
        let mut params = sys::llama_model_quantize_default_params();
        params.ftype = spec.ftype.value();
        params.nthread = nthread;
        params.allow_requantize = spec.allow_requantize;
        params.keep_split = job.keep_split;
        sys::llama_model_quantize(inp.as_ptr(), out.as_ptr(), &params)
    };
    if status != 0 {
        return Err(QuantError::Llama(format!(
            "llama_model_quantize returned {status} for {}; the reason is in llama.cpp's log on stderr",
            model.paths[0].display()
        )));
    }

    let written = Model::open(&job.outputs[0])
        .map_err(|e| QuantError::Llama(format!("it reported success, but {} does not open: {e}", job.outputs[0].display())))?;
    if written.paths != job.outputs {
        return Err(QuantError::Llama(format!(
            "expected {} output files and found {}",
            job.outputs.len(),
            written.paths.len()
        )));
    }
    let declared = written.meta().file_type();
    if declared != Some(spec.ftype.value()) {
        let found = declared.map_or_else(|| "none".to_owned(), |v| Ftype::from_file_type(v).map_or(v.to_string(), |f| f.to_string()));
        return Err(QuantError::Llama(format!(
            "{} declares general.file_type {found}, not {}",
            job.outputs[0].display(),
            spec.ftype
        )));
    }

    Ok(QuantReport {
        input: model.paths[0].clone(),
        output: job.outputs[0].clone(),
        from: job.from,
        to: spec.ftype,
        requantized: job.requantized,
        bytes_in: size_on_disk(&model.paths),
        bytes_out: size_on_disk(&job.outputs),
    })
}

/// Purpose: This function checks the output paths against the resolved files on disk and
/// creates the output directory.
///
/// Inputs:
/// - `model`: the opened input model.
/// - `outputs`: every file llama.cpp will write.
///
/// Outputs: `Ok(())`, or a [`QuantError::BadInput`].
fn check_outputs_on_disk(model: &Model, outputs: &[PathBuf]) -> Result<(), QuantError> {
    let inputs: Vec<PathBuf> = model.paths.iter().filter_map(|p| std::fs::canonicalize(p).ok()).collect();
    for out in outputs {
        if let Ok(resolved) = std::fs::canonicalize(out) {
            if inputs.contains(&resolved) {
                return Err(QuantError::BadInput(format!(
                    "the output {} is one of the input's own files; quantizing in place would destroy the input",
                    out.display()
                )));
            }
        }
        let dir = out.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
        std::fs::create_dir_all(dir)
            .map_err(|e| QuantError::BadInput(format!("cannot create the output directory {}: {e}", dir.display())))?;
        let resolved_dir = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
        // `in_hub_cache` judges a file by the directories above it, so ask about a file
        // in this directory.
        if in_hub_cache(&resolved_dir.join("x")) {
            return Err(QuantError::BadInput(cache_refusal(out)));
        }
    }
    Ok(())
}

/// Purpose: This function converts a file path into the C string llama.cpp takes.
///
/// Inputs:
/// - `path`: the file path.
///
/// Outputs: the C string, or an error when the path holds a NUL byte.
fn c_path(path: &Path) -> Result<CString, QuantError> {
    #[cfg(unix)]
    let bytes = {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes().to_vec()
    };
    #[cfg(not(unix))]
    let bytes = path
        .to_str()
        .ok_or_else(|| QuantError::BadInput(format!("{} is not valid UTF-8", path.display())))?
        .as_bytes()
        .to_vec();
    CString::new(bytes).map_err(|_| QuantError::BadInput(format!("{} contains a NUL byte", path.display())))
}

/// Purpose: This function adds up the sizes of a model's files on disk.
///
/// Inputs:
/// - `paths`: the model's files.
///
/// Outputs: the total bytes.
fn size_on_disk(paths: &[PathBuf]) -> u64 {
    paths.iter().filter_map(|p| std::fs::metadata(p).ok()).map(|m| m.len()).sum()
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;
    use crate::gguf::{ggml_type_layout, ggml_type_name};
    use crate::quantize::default_output;
    use crate::quantize::fixture::{scratch_dir, write_tiny_llama, write_tiny_llama_split};

    /// Purpose: This function lists the ggml type of every tensor in a model.
    ///
    /// Inputs:
    /// - `m`: the opened model.
    ///
    /// Outputs: each tensor's name and `ggml_type` number.
    fn types(m: &Model) -> Vec<(String, u32)> {
        m.tensors().map(|t| (t.info.name.clone(), t.info.kind)).collect()
    }

    #[test]
    fn quantize_fixture_to_q8_0() {
        let dir = scratch_dir("quantize-q8_0");
        let input = dir.join("tiny-llama-F32.gguf");
        write_tiny_llama(&input);
        let output = default_output(&input, Ftype::Q8_0);
        assert_eq!(output, dir.join("tiny-llama-F32.Q8_0.gguf"));

        let report = quantize(&input, &output, &QuantSpec::new(Ftype::Q8_0)).unwrap();
        assert_eq!(report.input, input);
        assert_eq!(report.output, output);
        assert_eq!(report.from, Some(Ftype::F32));
        assert_eq!(report.to, Ftype::Q8_0);
        assert!(!report.requantized);
        assert_eq!(report.bytes_in, std::fs::metadata(&input).unwrap().len());
        assert!(report.bytes_out < report.bytes_in / 3, "Q8_0 is about a quarter of F32: {report:?}");

        let out = Model::open(&output).unwrap();
        assert_eq!(out.meta().file_type(), Some(Ftype::Q8_0.value()));
        for (name, kind) in types(&out) {
            let want = if name.ends_with("norm.weight") { "F32" } else { "Q8_0" };
            assert_eq!(ggml_type_name(kind), want, "{name}");
        }

        // The output is quantized now, so going again needs the flag, and with it the
        // report says so.
        let again = dir.join("again.gguf");
        let err = quantize(&output, &again, &QuantSpec::new(Ftype::Q4_0)).unwrap_err();
        assert!(matches!(err, QuantError::Requantize { .. }), "{err:?}");
        let report = quantize(&output, &again, &QuantSpec { allow_requantize: true, ..QuantSpec::new(Ftype::Q4_0) }).unwrap();
        assert!(report.requantized);
        assert_eq!(report.from, Some(Ftype::Q8_0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_split_input_is_read_from_its_first_part() {
        let dir = scratch_dir("quantize-split");
        let parts = write_tiny_llama_split(&dir, "tiny-F32");
        let output = default_output(&parts[0], Ftype::Q8_0);
        assert_eq!(output, dir.join("tiny-F32.Q8_0.gguf"));

        // keep_split is on by default, so two parts in give two parts out.
        let report = quantize(&parts[0], &output, &QuantSpec::new(Ftype::Q8_0)).unwrap();
        assert_eq!(report.output, dir.join("tiny-F32.Q8_0-00001-of-00002.gguf"));
        let kept = Model::open(&report.output).unwrap();
        assert_eq!(kept.part_count(), 2);
        let input = Model::open(&parts[0]).unwrap();
        assert_eq!(kept.part(1).tensors.len(), input.part(1).tensors.len(), "each tensor stays in its part");
        assert_eq!(report.bytes_in, std::fs::metadata(&parts[0]).unwrap().len() + std::fs::metadata(&parts[1]).unwrap().len());

        // Without keep_split, llama.cpp writes one file holding every tensor.
        let one = dir.join("one.gguf");
        let spec = QuantSpec { keep_split: false, ..QuantSpec::new(Ftype::Q8_0) };
        let report = quantize(&parts[0], &one, &spec).unwrap();
        let merged = Model::open(&report.output).unwrap();
        assert_eq!(merged.part_count(), 1);
        assert_eq!(merged.tensors().count(), input.tensors().count());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_fixture_loads_in_llama_cpp() {
        // The quantizer only reads hparams, so it would accept a fixture with wrong
        // shapes. A full load checks every tensor against the architecture.
        let dir = scratch_dir("quantize-load");
        let path = dir.join("tiny.gguf");
        write_tiny_llama(&path);
        let c = c_path(&path).unwrap();
        // SAFETY: a test of the fixture, not library code. `c` outlives the call, the
        // params come from llama.cpp's default constructor with only plain fields
        // changed, and a non-null model is freed exactly once.
        let loaded = unsafe {
            BACKEND_INIT.call_once(|| sys::llama_backend_init());
            let mut params = sys::llama_model_default_params();
            params.n_gpu_layers = 0;
            let model = sys::llama_model_load_from_file(c.as_ptr(), params);
            let n = if model.is_null() { None } else { Some(sys::llama_model_n_params(model)) };
            if !model.is_null() {
                sys::llama_model_free(model);
            }
            n
        };
        let expected: u64 = crate::quantize::fixture::tensor_shapes().iter().map(|(_, d)| d.iter().product::<u64>()).sum();
        assert_eq!(loaded, Some(expected), "llama.cpp refused the fixture; its log above says why");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn type_table_matches_ggml() {
        let mut checked = 0;
        for t in 0..sys::GGML_TYPE_COUNT {
            // SAFETY: ggml's type-trait getters read a static table, and `t` is below
            // GGML_TYPE_COUNT, the bound they assert.
            let (blck, size, name, quantized) = unsafe {
                (
                    sys::ggml_blck_size(t),
                    sys::ggml_type_size(t),
                    CStr::from_ptr(sys::ggml_type_name(t)).to_string_lossy().into_owned(),
                    sys::ggml_is_quantized(t),
                )
            };
            let ours = ggml_type_name(t);
            if ours == "type ?" {
                // A retired tag: ggml keeps a placeholder entry with no size.
                assert_eq!(blck, 0, "ggml type {t} ({name}) has a size, but dovata has no name for it");
                continue;
            }
            assert!(ours.eq_ignore_ascii_case(&name), "type {t}: dovata says {ours}, ggml says {name}");
            let layout = ggml_type_layout(t).unwrap();
            assert_eq!(layout, (blck as u64, size as u64), "type {t} ({ours}): (elements, bytes) per block");
            assert_eq!(super::super::is_quantized_type(t), quantized, "type {t} ({ours}): quantized");
            checked += 1;
        }
        assert!(checked >= 30, "only {checked} types checked");
    }
}
