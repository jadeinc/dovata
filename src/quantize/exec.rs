//! Author: Joe Ricks
//! File: src/quantize/exec.rs
//! Description: Quantizes a GGUF model by running llama.cpp's `llama-quantize` program.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::{cache_refusal, in_hub_cache, prepare, Ftype, Job, QuantError, QuantReport, QuantSpec};
use crate::load::Model;

/// The program dovata runs to quantize, found on `PATH`.
pub(crate) const PROGRAM: &str = "llama-quantize";

/// How many lines of the program's log a failure message quotes.
const LOG_TAIL: usize = 20;

/// Purpose: This function writes a quantized copy of a GGUF model by running
/// `llama-quantize`.
///
/// Inputs:
/// - `input`: the model file, or the first part of a split model.
/// - `output`: the file to write, or for a split output the name its parts are made from.
/// - `spec`: the file type to make and the options.
///
/// Outputs: a [`QuantReport`] of what was written, or a [`QuantError`].
pub fn quantize(
    input: &Path,
    output: &Path,
    spec: &QuantSpec,
) -> Result<QuantReport, QuantError>
{
    let model = Model::open(input).map_err(|e| QuantError::BadInput(format!("cannot open the input: {e}")))?;
    let job = prepare(&model, output, spec)?;
    check_outputs_on_disk(&model, &job.outputs)?;
    let program = find_program().ok_or(QuantError::NotInstalled)?;

    let args = arguments(&model.paths[0], &job, spec)?;
    run(&program, &args).map_err(|tail| {
        QuantError::Llama(format!("{PROGRAM} failed for {}. The end of its log:\n{tail}", model.paths[0].display()))
    })?;

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

/// Purpose: This function finds `llama-quantize` on `PATH`.
///
/// Inputs: none.
///
/// Outputs: the program's path, or `None` when it is not installed.
pub(crate) fn find_program() -> Option<PathBuf>
{
    let names: &[&str] = if cfg!(windows) { &["llama-quantize.exe"] } else { &[PROGRAM] };
    std::env::split_paths(&std::env::var_os("PATH")?)
        .flat_map(|dir| names.iter().map(move |n| dir.join(n)))
        .find(|p| p.is_file())
}

/// Purpose: This function builds the command-line arguments for `llama-quantize`.
///
/// Inputs:
/// - `input`: the input model's first file.
/// - `job`: the checked request, with the output name.
/// - `spec`: the file type and options.
///
/// Outputs: the arguments, or an error when a path cannot be made absolute.
fn arguments(
    input: &Path,
    job: &Job,
    spec: &QuantSpec,
) -> Result<Vec<OsString>, QuantError>
{
    let absolute = |p: &Path| {
        std::path::absolute(p).map_err(|e| QuantError::BadInput(format!("cannot make {} absolute: {e}", p.display())))
    };
    let mut args: Vec<OsString> = Vec::new();
    if spec.allow_requantize {
        args.push("--allow-requantize".into());
    }
    // For a split output the program takes `<prefix>.gguf` and names the parts
    // `<prefix>-00001-of-0000N.gguf` itself.
    let output = if job.keep_split {
        args.push("--keep-split".into());
        let mut prefix = absolute(&job.target)?.into_os_string();
        prefix.push(".gguf");
        prefix
    } else {
        absolute(&job.target)?.into_os_string()
    };
    args.push(absolute(input)?.into_os_string());
    args.push(output);
    args.push(spec.ftype.name().into());
    if spec.nthread > 0 {
        args.push(spec.nthread.to_string().into());
    }
    Ok(args)
}

/// Purpose: This function runs `llama-quantize`, showing its log on standard error as it
/// runs.
///
/// Inputs:
/// - `program`: the program's path.
/// - `args`: its arguments.
///
/// Outputs: `Ok(())` when it succeeds; otherwise the exit status and the last lines of
/// its log, as text.
fn run(
    program: &Path,
    args: &[OsString],
) -> Result<(), String>
{
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start {}: {e}", program.display()))?;
    let stdout = child.stdout.take().expect("stdout is piped");
    let stderr = child.stderr.take().expect("stderr is piped");
    let out = std::thread::spawn(move || echo(stdout));
    let err_tail = echo(stderr);
    let out_tail = out.join().unwrap_or_default();
    let status = child.wait().map_err(|e| format!("cannot wait for {}: {e}", program.display()))?;
    if status.success() {
        return Ok(());
    }
    // llama.cpp logs to standard error; a usage message goes to standard output.
    let tail = if err_tail.is_empty() { out_tail } else { err_tail };
    Err(format!("{status}\n{}", tail.join("\n")))
}

/// Purpose: This function copies a program's output to standard error, line by line,
/// and keeps the last lines.
///
/// Inputs:
/// - `from`: the program's standard output or standard error.
///
/// Outputs: up to [`LOG_TAIL`] of the last lines that are not blank.
fn echo(
    from: impl Read,
) -> Vec<String>
{
    let mut tail = VecDeque::with_capacity(LOG_TAIL);
    for line in BufReader::new(from).split(b'\n') {
        let Ok(line) = line else { break };
        let line = String::from_utf8_lossy(&line).into_owned();
        eprintln!("{line}");
        if line.trim().is_empty() {
            continue;
        }
        if tail.len() == LOG_TAIL {
            tail.pop_front();
        }
        tail.push_back(line);
    }
    tail.into()
}

/// Purpose: This function checks the output paths against the resolved files on disk and
/// creates the output directory.
///
/// Inputs:
/// - `model`: the opened input model.
/// - `outputs`: every file llama.cpp will write.
///
/// Outputs: `Ok(())`, or a [`QuantError::BadInput`].
fn check_outputs_on_disk(
    model: &Model,
    outputs: &[PathBuf],
) -> Result<(), QuantError>
{
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

/// Purpose: This function adds up the sizes of a model's files on disk.
///
/// Inputs:
/// - `paths`: the model's files.
///
/// Outputs: the total bytes.
fn size_on_disk(
    paths: &[PathBuf],
) -> u64
{
    paths.iter().filter_map(|p| std::fs::metadata(p).ok()).map(|m| m.len()).sum()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::gguf::ggml_type_name;
    use crate::quantize::default_output;
    use crate::quantize::fixture::{scratch_dir, write_tiny_llama, write_tiny_llama_split};

    /// Purpose: This function checks that `llama-quantize` is installed before a test
    /// that runs it.
    ///
    /// Inputs:
    /// - `test`: the test's name, for the message printed when it is skipped.
    ///
    /// Outputs: `true` when the program is on `PATH`; otherwise it prints that the test is
    /// skipped and returns `false`.
    fn installed(
        test: &str,
    ) -> bool
    {
        if find_program().is_some() {
            return true;
        }
        eprintln!("skipping {test}: {PROGRAM} is not on PATH. Install llama.cpp to run it");
        false
    }

    /// Purpose: This function lists the ggml type of every tensor in a model.
    ///
    /// Inputs:
    /// - `m`: the opened model.
    ///
    /// Outputs: each tensor's name and `ggml_type` number.
    fn types(
        m: &Model,
    ) -> Vec<(String, u32)>
    {
        m.tensors().map(|t| (t.info.name.clone(), t.info.kind)).collect()
    }

    /// Purpose: This function builds a checked quantization request without opening a
    /// model.
    ///
    /// Inputs:
    /// - `target`: the output name the program is given.
    /// - `keep_split`: whether the output is split.
    ///
    /// Outputs: the [`Job`].
    fn job(
        target: &str,
        keep_split: bool,
    ) -> Job
    {
        Job { from: None, requantized: false, keep_split, target: target.into(), outputs: vec![target.into()] }
    }

    #[test]
    fn arguments_follow_llama_quantize_usage()
    {
        let spec = QuantSpec::new(Ftype::Q8_0);
        let args = arguments(Path::new("/in/m.gguf"), &job("/o/m.Q8_0.gguf", false), &spec).unwrap();
        assert_eq!(args, ["/in/m.gguf", "/o/m.Q8_0.gguf", "Q8_0"]);

        let spec = QuantSpec { allow_requantize: true, nthread: 4, ..QuantSpec::new(Ftype::Q4_K_M) };
        let args = arguments(Path::new("/in/m-00001-of-00002.gguf"), &job("/o/m.Q4_K_M", true), &spec).unwrap();
        assert_eq!(
            args,
            ["--allow-requantize", "--keep-split", "/in/m-00001-of-00002.gguf", "/o/m.Q4_K_M.gguf", "Q4_K_M", "4"],
            "a split output is the prefix plus .gguf, which the program strips"
        );

        // llama-quantize reads an output that starts with a digit, such as 7b.gguf, as a
        // type number, so every path it gets is absolute.
        let args = arguments(Path::new("7b.gguf"), &job("7b.Q8_0.gguf", false), &QuantSpec::new(Ftype::Q8_0)).unwrap();
        assert!(args[..2].iter().all(|a| Path::new(a).is_absolute()), "{args:?}");
    }

    #[test]
    fn echo_keeps_the_last_lines_of_a_log()
    {
        let log: String = (1..=30).map(|i| format!("line {i}\n\n")).collect();
        let tail = echo(log.as_bytes());
        assert_eq!(tail.len(), LOG_TAIL);
        assert_eq!((tail[0].as_str(), tail[LOG_TAIL - 1].as_str()), ("line 11", "line 30"), "blank lines are dropped");
        assert_eq!(echo(&b"no newline at the end"[..]), ["no newline at the end"]);
    }

    #[test]
    #[cfg(unix)]
    fn a_failing_program_reports_the_end_of_its_log()
    {
        // `sh` stands in for llama-quantize: it logs to stderr and exits 3.
        let args: Vec<OsString> = ["-c", "echo working >&2; echo 'no tensor type fallback' >&2; exit 3"].map(Into::into).into();
        let err = run(Path::new("/bin/sh"), &args).unwrap_err();
        assert!(err.contains('3') && err.ends_with("working\nno tensor type fallback"), "{err}");
        assert_eq!(run(Path::new("/bin/sh"), &["-c".into(), "exit 0".into()]), Ok(()));
    }

    #[test]
    fn quantize_fixture_to_q8_0()
    {
        if !installed("quantize_fixture_to_q8_0") {
            return;
        }
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
    fn a_split_input_is_read_from_its_first_part()
    {
        if !installed("a_split_input_is_read_from_its_first_part") {
            return;
        }
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
    fn type_sizes_match_what_llama_quantize_writes()
    {
        // llama-quantize pads each tensor's data to the file's alignment, so if dovata's
        // size for a ggml type were wrong, the padded end of a tensor would miss the next
        // tensor's offset or the end of the file. The fixture's rows are 64 and 128 wide,
        // so llama.cpp falls back to 32-wide blocks wherever a type needs 256. The
        // 256-wide sizes are checked against real headers by
        // a_real_headers_tensors_sit_end_to_end in src/load/model.rs.
        if !installed("type_sizes_match_what_llama_quantize_writes") {
            return;
        }
        let dir = scratch_dir("quantize-sizes");
        let input = dir.join("tiny-F32.gguf");
        write_tiny_llama(&input);
        let mut seen = BTreeSet::new();
        // Q1_0 is left out: it needs rows a multiple of 128 wide, and llama.cpp has no
        // fallback type for it, so it refuses the 64-wide fixture.
        let testable = |f: &Ftype| f.is_target() && !f.needs_imatrix() && !matches!(f, Ftype::F32 | Ftype::Q1_0);
        for f in Ftype::ALL.into_iter().filter(testable) {
            let output = dir.join(format!("tiny.{}.gguf", f.name()));
            quantize(&input, &output, &QuantSpec::new(f)).unwrap_or_else(|e| panic!("{f}: {e}"));
            let m = Model::open(&output).unwrap();
            let (align, data) = (m.meta().alignment, m.meta().data_offset.unwrap());
            let file_len = std::fs::metadata(&output).unwrap().len();
            let mut tensors: Vec<_> = m.tensors().collect();
            tensors.sort_by_key(|t| t.file_offset);
            for (i, t) in tensors.iter().enumerate() {
                let bytes = t.info.bytes().unwrap_or_else(|| panic!("{f}: dovata has no size for ggml type {}", t.info.kind));
                let padded_end = data + (t.file_offset - data + bytes).div_ceil(align) * align;
                let next = tensors.get(i + 1).map_or(file_len, |n| n.file_offset);
                assert_eq!(padded_end, next, "{f}: {} is {}, and dovata's size for it is wrong", t.info.name, ggml_type_name(t.info.kind));
                seen.insert(ggml_type_name(t.info.kind));
            }
        }
        for want in ["F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"] {
            assert!(seen.contains(want), "no output had a {want} tensor: {seen:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ftype_table_matches_llama_quantize_help()
    {
        if !installed("ftype_table_matches_llama_quantize_help") {
            return;
        }
        // `--help` lists each type it writes as `  15  or  Q4_K_M  : ...`, aliases such as
        // Q4_K included. COPY has no number: it is a mode, not a type.
        let help = Command::new(find_program().unwrap()).arg("--help").output().unwrap();
        let text = String::from_utf8_lossy(&help.stdout);
        let offered: Vec<(u32, String)> = text
            .lines()
            .filter_map(|l| {
                let (number, rest) = l.trim().split_once("  or  ")?;
                Some((number.trim().parse().ok()?, rest.split_whitespace().next()?.to_owned()))
            })
            .collect();
        assert!(offered.len() > 30, "read only {} types from --help:\n{text}", offered.len());
        for (value, name) in &offered {
            let f = Ftype::parse(name).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(f.value(), *value, "{name}");
        }
        for f in Ftype::ALL {
            let listed = offered.iter().any(|(v, n)| *v == f.value() && n == f.name());
            assert_eq!(listed, f.is_target(), "{f}: listed by {PROGRAM} --help");
        }
    }
}
