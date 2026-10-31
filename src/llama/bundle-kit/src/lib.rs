//! Author: Joe Ricks
//! File: src/llama/bundle-kit/src/lib.rs
//! Description: Reads and checks the patch bundles that customize llama.cpp at build time.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Paths a patch must never touch, and why.
///
/// The engine links *upstream plus these patches*; the quantizer
/// (`tools/scripts/quantize-model.py`) builds tools from *stock upstream* at the same
/// pin. That is only sound while no patch changes how a GGUF is produced or read.
/// A patch that edited the converter would make the engine and the quantizer disagree
/// about the file format, and the symptom is a published `model.gguf` that the engine
/// refuses — after an hours-long build on a GPU box.
///
/// This is the "one pin, not two" invariant from CLAUDE.md, expressed as a rule a
/// machine can check.
pub const FORBIDDEN_PATH_PREFIXES: &[&str] = &[
    "convert_hf_to_gguf.py",
    "convert_hf_to_gguf_update.py",
    "gguf-py/",
    "tools/quantize/",
    "src/llama-quant.cpp",
];

/// Purpose: This struct holds one bundle's patches and cmake defines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bundle {
    /// Capability name — matches the cargo feature that enables it.
    pub name: String,
    /// Directory holding `bundle.toml`.
    pub dir: PathBuf,
    /// Upstream commit the series was generated against.
    pub upstream: String,
    /// Absolute paths to the patch files, in application order.
    pub patches: Vec<PathBuf>,
    /// `-D` arguments for cmake, including the resolved source directory.
    pub defines: Vec<(String, String)>,
}

/// Purpose: This struct holds the raw contents of a `bundle.toml`.
#[derive(Deserialize)]
struct Manifest {
    name: String,
    upstream: String,
    patches: Vec<String>,
    #[serde(default)]
    defines: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    sources: Sources,
}

/// Purpose: This struct holds the `[sources]` table of a `bundle.toml`.
#[derive(Deserialize, Default)]
struct Sources {
    driver_dir: Option<String>,
}

/// Purpose: This function reads and checks the `bundle.toml` in a bundle directory.
///
/// Inputs:
/// - `dir`: the bundle directory.
///
/// Outputs: the [`Bundle`], or an error message.
///
/// # Errors
///
/// Returns an error if `bundle.toml` cannot be read, is not valid, lists no patches, or
/// names a patch file that does not exist.
pub fn load_manifest(dir: &Path) -> Result<Bundle, String> {
    let path = dir.join("bundle.toml");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("reading {}: {e}", path.display()))?;
    parse_manifest(dir, &text)
}

/// Purpose: This function parses the text of a `bundle.toml` into a [`Bundle`].
///
/// Inputs:
/// - `dir`: the bundle directory, where the patch files must be.
/// - `text`: the contents of `bundle.toml`.
///
/// Outputs: the [`Bundle`], or an error message.
///
/// # Errors
///
/// Returns an error if the text is not valid TOML for a bundle, lists no patches, or
/// names a patch file that is not in `dir`.
pub fn parse_manifest(dir: &Path, text: &str) -> Result<Bundle, String> {
    let m: Manifest = toml::from_str(text).map_err(|e| format!("{}/bundle.toml: {e}", dir.display()))?;

    if m.patches.is_empty() {
        return Err(format!(
            "{}/bundle.toml lists no patches. A bundle with no delta is not a bundle — \
             delete it, or the build will look like it customized llama.cpp and did not.",
            dir.display()
        ));
    }

    let mut patches = Vec::with_capacity(m.patches.len());
    for rel in &m.patches {
        let p = dir.join(rel);
        if !p.is_file() {
            return Err(format!(
                "{}/bundle.toml names patch '{rel}', which does not exist. Refusing rather \
                 than skipping: a partly-applied series builds cleanly and runs stock.",
                dir.display()
            ));
        }
        patches.push(p);
    }

    let mut defines: Vec<(String, String)> = m.defines.into_iter().collect();
    if let Some(rel) = m.sources.driver_dir {
        // Absolute, because the patched CMakeLists is read from a staged copy of the
        // tree whose depth relative to this directory is not knowable here — that
        // counted-relative-path fragility is what the bundle exists to remove.
        let abs = dir.join(rel);
        let abs = abs.canonicalize().unwrap_or(abs);
        defines.push((
            format!("JADE_{}_DRIVER_DIR", m.name.to_uppercase()),
            abs.display().to_string(),
        ));
    }

    Ok(Bundle { name: m.name, dir: dir.to_path_buf(), upstream: m.upstream, patches, defines })
}

/// Purpose: This function finds the first forbidden file a patch edits.
///
/// Inputs:
/// - `patch_text`: the patch, as a unified diff.
///
/// Outputs: the forbidden path and the rule it matches, or `None`.
#[must_use]
pub fn patch_touches_forbidden_path(patch_text: &str) -> Option<String> {
    for line in patch_text.lines() {
        let Some(rest) = line.strip_prefix("+++ ") else { continue };
        let path = rest.trim().trim_start_matches("b/");
        if path == "/dev/null" {
            continue;
        }
        if let Some(bad) = FORBIDDEN_PATH_PREFIXES.iter().find(|p| path.starts_with(**p)) {
            return Some(format!("{path} (matches '{bad}')"));
        }
    }
    None
}

/// Purpose: This function checks that no patch in the bundles edits a forbidden file.
///
/// Inputs:
/// - `bundles`: the bundles to check.
///
/// Outputs: `Ok(())`, or an error message.
///
/// # Errors
///
/// Returns an error naming the bundle, the patch and the file if a patch edits a path in
/// [`FORBIDDEN_PATH_PREFIXES`], or if a patch file cannot be read.
pub fn check_forbidden_paths(bundles: &[Bundle]) -> Result<(), String> {
    for b in bundles {
        for p in &b.patches {
            let text = std::fs::read_to_string(p).map_err(|e| format!("reading {}: {e}", p.display()))?;
            if let Some(bad) = patch_touches_forbidden_path(&text) {
                return Err(format!(
                    "bundle '{}' patch {} edits {bad}. Patches must not change how a GGUF is \
                     produced or read: the engine links upstream+patches while the quantizer \
                     builds its tools from stock upstream at the same pin, and the two agreeing \
                     is what makes one pin enough.",
                    b.name,
                    p.file_name().unwrap_or(p.as_os_str()).to_string_lossy(),
                ));
            }
        }
    }
    Ok(())
}

/// Purpose: This function merges the bundles' cmake defines into one list.
///
/// Inputs:
/// - `bundles`: the enabled bundles.
///
/// Outputs: each define once, in first-seen order, or an error message.
///
/// # Errors
///
/// Returns an error if two bundles set the same define to different values.
pub fn cmake_defines(bundles: &[Bundle]) -> Result<Vec<(String, String)>, String> {
    let mut out: Vec<(String, String)> = Vec::new();
    for b in bundles {
        for (k, v) in &b.defines {
            if let Some((_, existing)) = out.iter().find(|(ek, _)| ek == k) {
                if existing != v {
                    return Err(format!(
                        "bundles disagree on cmake define {k}: '{existing}' vs '{v}' (from '{}')",
                        b.name
                    ));
                }
                continue;
            }
            out.push((k.clone(), v.clone()));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Purpose: This function creates an empty temporary folder for one test.
    ///
    /// Inputs:
    /// - `tag`: a name that keeps each test's folder separate.
    ///
    /// Outputs: the folder's path.
    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bundle-kit-test-{tag}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    const MANIFEST: &str = r#"
name = "moe"
upstream = "deadbeef"
patches = ["patches/0001-x.patch"]
[defines]
JADE_MOE_DRIVER = "ON"
[sources]
driver_dir = ".."
"#;

    #[test]
    fn reads_name_upstream_patches_and_defines() {
        let d = tmpdir("ok");
        std::fs::create_dir_all(d.join("patches")).unwrap();
        std::fs::write(d.join("patches/0001-x.patch"), "").unwrap();
        let b = parse_manifest(&d, MANIFEST).unwrap();
        assert_eq!(b.name, "moe");
        assert_eq!(b.upstream, "deadbeef");
        assert_eq!(b.patches.len(), 1);
        assert!(b.defines.iter().any(|(k, v)| k == "JADE_MOE_DRIVER" && v == "ON"));
        // driver_dir becomes an absolute -D so the staged tree's depth cannot matter.
        let (k, v) = b.defines.iter().find(|(k, _)| k == "JADE_MOE_DRIVER_DIR").unwrap();
        assert_eq!(k, "JADE_MOE_DRIVER_DIR");
        assert!(Path::new(v).is_absolute(), "driver dir must be absolute, got {v}");
    }

    /// A named-but-missing patch is an error, not a skip. Skipping is how you get a
    /// build with half a seam in it that reports no problem at all.
    #[test]
    fn missing_patch_file_is_an_error() {
        let d = tmpdir("missing");
        let err = parse_manifest(&d, MANIFEST).unwrap_err();
        assert!(err.contains("0001-x.patch"), "must name the missing patch; got: {err}");
    }

    #[test]
    fn empty_patch_list_is_an_error() {
        let d = tmpdir("empty");
        let err = parse_manifest(&d, "name = \"x\"\nupstream = \"a\"\npatches = []\n").unwrap_err();
        assert!(err.contains("no patches"), "got: {err}");
    }

    #[test]
    fn rejects_a_patch_touching_the_gguf_converter() {
        let diff = "diff --git a/convert_hf_to_gguf.py b/convert_hf_to_gguf.py\n\
                    --- a/convert_hf_to_gguf.py\n+++ b/convert_hf_to_gguf.py\n@@ -1 +1 @@\n-a\n+b\n";
        let hit = patch_touches_forbidden_path(diff).expect("must be rejected");
        assert!(hit.contains("convert_hf_to_gguf.py"), "got: {hit}");
    }

    #[test]
    fn rejects_a_patch_touching_gguf_py() {
        let diff = "+++ b/gguf-py/gguf/constants.py\n";
        assert!(patch_touches_forbidden_path(diff).is_some());
    }

    #[test]
    fn rejects_a_patch_touching_the_quantizer() {
        assert!(patch_touches_forbidden_path("+++ b/src/llama-quant.cpp\n").is_some());
        assert!(patch_touches_forbidden_path("+++ b/tools/quantize/quantize.cpp\n").is_some());
    }

    /// The real MoE seam must pass, or the rule is too broad to ship.
    #[test]
    fn allows_the_seam_paths_the_moe_bundle_actually_touches() {
        for p in [
            "+++ b/src/llama-graph.cpp\n",
            "+++ b/ggml/src/ggml-cuda/ggml-cuda.cu\n",
            "+++ b/ggml/src/ggml-cuda/CMakeLists.txt\n",
            "+++ b/src/CMakeLists.txt\n",
        ] {
            assert!(patch_touches_forbidden_path(p).is_none(), "wrongly rejected {p}");
        }
    }

    /// `+++ /dev/null` is a deletion, not a path being edited; it must not be matched
    /// against the prefix list (and must not panic).
    #[test]
    fn ignores_dev_null_targets() {
        assert!(patch_touches_forbidden_path("+++ /dev/null\n").is_none());
    }

    #[test]
    fn no_bundles_means_no_defines() {
        assert!(cmake_defines(&[]).unwrap().is_empty());
    }

    #[test]
    fn conflicting_defines_are_an_error_rather_than_last_wins() {
        let mk = |name: &str, val: &str| Bundle {
            name: name.into(),
            dir: PathBuf::from("/x"),
            upstream: "a".into(),
            patches: vec![],
            defines: vec![("SHARED".to_string(), val.to_string())],
        };
        let err = cmake_defines(&[mk("a", "ON"), mk("b", "OFF")]).unwrap_err();
        assert!(err.contains("SHARED"), "must name the define; got: {err}");
    }

    /// Every bundle that actually ships must load, and must not name a patch it does
    /// not have. Everything above tests the rules against synthetic input; this is the
    /// one test that would notice a typo in a real `bundle.toml`, and it runs on a
    /// laptop with no llama.cpp and no GPU.
    ///
    /// It *discovers* the bundles rather than naming one. An earlier version reached for
    /// `../moe-driver/bundle` and broke the moment that directory moved to live with the
    /// capability that owns it — a counted relative path, which is the exact fragility
    /// this crate exists to remove from the CMake. Discovery also means a second
    /// capability's bundle is covered the day it appears.
    #[test]
    fn every_shipped_bundle_loads_and_is_allowed() {
        // <root>/src/llama/bundle-kit -> <root>
        let src = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .expect("manifest dir is <root>/src/llama/bundle-kit")
            .join("src");

        let mut found = Vec::new();
        let mut stack = vec![src];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    // Skip the upstream checkout: it is large, and none of it is ours.
                    if p.file_name().is_some_and(|n| n == "llama.cpp") {
                        continue;
                    }
                    stack.push(p);
                } else if p.file_name().is_some_and(|n| n == "bundle.toml") {
                    found.push(dir.clone());
                }
            }
        }

        // Finding none is valid: dovata 3.0 removed the only bundle, the runtime MoE
        // pager's. This used to assert at least one, to catch a bundle that moved out of
        // reach of the walk; with none shipped, that check has nothing to protect.
        for dir in &found {
            let b = load_manifest(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
            assert_eq!(
                b.upstream.len(),
                40,
                "{}: upstream must be a full sha, not a branch or tag",
                dir.display()
            );
            check_forbidden_paths(std::slice::from_ref(&b))
                .unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        }
    }

    #[test]
    fn identical_defines_from_two_bundles_collapse() {
        let mk = |name: &str| Bundle {
            name: name.into(),
            dir: PathBuf::from("/x"),
            upstream: "a".into(),
            patches: vec![],
            defines: vec![("SHARED".to_string(), "ON".to_string())],
        };
        assert_eq!(cmake_defines(&[mk("a"), mk("b")]).unwrap().len(), 1);
    }
}
