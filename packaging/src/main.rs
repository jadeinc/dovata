//! packaging — build the ovata daemon + provider `.so` packages and assemble a `dist/`
//! bundle. This is the single source of truth for how a release is built and staged;
//! CI (`.github/workflows/release.yml`) is a thin wrapper that sets up the environment
//! and calls into here.
//!
//! Usage:
//!   cargo run -p packaging -- build [--release]     # dev bundle (daemon copied as `dovata`)
//!   cargo run -p packaging -- release [--cuda]      # release bundle: `dovata-<arch>` + sha256
//!
//! Both modes build every remote provider cdylib (anthropic/openai/echo), read each
//! one's advertised name + provider-ABI version through its exported C symbols (which
//! also validates the `.so` is a real `export_provider!` plugin), copy the `.so` into
//! `dist/`, and write `dist/manifest.json`. The local `dovata` engine is compiled *into*
//! the daemon (it's not a `.so`), so the daemon binary is staged as-is.
//!
//! `release` builds the release profile, stages the daemon under its canonical,
//! bootloader-facing asset name `dovata-<arch>` with a `.sha256` sidecar, and (with
//! `--cuda`) builds the daemon's GPU backend. Signing / config bundles come later.

use std::path::{Path, PathBuf};
use std::process::Command;

use ovata_infer_protocol::PROVIDER_ABI_VERSION;
use sha2::{Digest, Sha256};

/// The remote provider crates that ship as loadable `.so` packages.
const PROVIDER_CRATES: &[&str] = &[
    // anthropic migrated to a pure-Jade library (anthropic/anthropic.jde); it is no
    // longer a native .so, so it is not packaged here. openai is mid-migration.
    "ovata-provider-openai",
    "ovata-provider-echo",
];
/// The daemon binary (host + the in-process local llama.cpp engine).
const DAEMON_CRATE: &str = "dovata";

/// How the bundle is built and how the daemon binary is staged.
struct Plan {
    /// Release profile (`target/release`) vs dev (`target/debug`).
    release: bool,
    /// Build the daemon's `cuda` feature (GPU offload). Release/Linux only.
    cuda: bool,
    /// Stage the daemon under its canonical `dovata-<arch>` asset name + a `.sha256`
    /// sidecar (release), rather than copying it as plain `dovata` (dev).
    stage_asset: bool,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    let plan = match args.first().map(String::as_str) {
        Some("build") => Plan { release: has("--release") || has("-r"), cuda: false, stage_asset: false },
        Some("release") => Plan { release: true, cuda: has("--cuda"), stage_asset: true },
        _ => {
            eprintln!("usage: cargo run -p packaging -- <build [--release] | release [--cuda]>");
            std::process::exit(2);
        }
    };
    if let Err(e) = assemble(&plan) {
        eprintln!("packaging: error: {e}");
        std::process::exit(1);
    }
}

fn assemble(plan: &Plan) -> Result<(), String> {
    let root = workspace_root();
    let profile = if plan.release { "release" } else { "debug" };
    let target_dir = root.join("target").join(profile);
    let dist = root.join("dist");

    // 1. Build the daemon + every provider cdylib in one cargo invocation.
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut cmd = Command::new(&cargo);
    cmd.arg("build").current_dir(&root);
    if plan.release {
        cmd.arg("--release");
    }
    cmd.args(["-p", DAEMON_CRATE]);
    if plan.cuda {
        cmd.args(["--features", "cuda"]);
    }
    for c in PROVIDER_CRATES {
        cmd.args(["-p", c]);
    }
    eprintln!(
        "packaging: building {DAEMON_CRATE}{} + {} providers ({profile})",
        if plan.cuda { " (cuda)" } else { "" },
        PROVIDER_CRATES.len()
    );
    let status = cmd.status().map_err(|e| format!("running cargo: {e}"))?;
    if !status.success() {
        return Err("cargo build failed".into());
    }

    // 2. Fresh dist/.
    if dist.exists() {
        std::fs::remove_dir_all(&dist).map_err(|e| format!("clearing {}: {e}", dist.display()))?;
    }
    std::fs::create_dir_all(&dist).map_err(|e| format!("creating {}: {e}", dist.display()))?;

    // 3. Collect each provider .so, reading its advertised name + ABI to validate it.
    let mut providers = Vec::new();
    for crate_name in PROVIDER_CRATES {
        let lib_name = crate_name.replace('-', "_");
        let file_name = libloading::library_filename(&lib_name);
        let src = target_dir.join(&file_name);
        if !src.exists() {
            return Err(format!("expected built library missing: {}", src.display()));
        }

        // SAFETY: reads only the zero-arg name/abi symbols; does not call `new`.
        let (name, abi) = unsafe { read_provider_meta(&src) }
            .map_err(|e| format!("{crate_name}: not a valid provider .so: {e}"))?;
        if abi != PROVIDER_ABI_VERSION {
            eprintln!(
                "packaging: WARNING {crate_name} advertises ABI {abi} != host ABI {PROVIDER_ABI_VERSION}"
            );
        }

        let dest = dist.join(&file_name);
        std::fs::copy(&src, &dest).map_err(|e| format!("copying {}: {e}", src.display()))?;
        eprintln!("packaging:   provider '{name}' (abi {abi}) -> dist/{}", file_name.to_string_lossy());
        providers.push(serde_json::json!({
            "crate": crate_name,
            "name": name,
            "abi_version": abi,
            "file": file_name.to_string_lossy(),
        }));
    }

    // 4. Stage the daemon binary.
    let daemon_src = target_dir.join(DAEMON_CRATE);
    let daemon = if !daemon_src.exists() {
        // The daemon builds on any platform, so this is unexpected — warn, don't fail,
        // so provider packaging still succeeds.
        eprintln!("packaging: WARNING daemon binary not found at {}", daemon_src.display());
        serde_json::Value::Null
    } else if plan.stage_asset {
        // Release: canonical, bootloader-facing asset name + a sha256 sidecar.
        let asset = format!("{DAEMON_CRATE}-{}", std::env::consts::ARCH);
        let dest = dist.join(&asset);
        std::fs::copy(&daemon_src, &dest).map_err(|e| format!("staging daemon: {e}"))?;
        let sha = sha256_file(&dest)?;
        let sha_file = format!("{asset}.sha256");
        std::fs::write(dist.join(&sha_file), format!("{sha}  {asset}\n"))
            .map_err(|e| format!("writing {sha_file}: {e}"))?;
        eprintln!("packaging:   daemon -> dist/{asset} (+ .sha256)");
        serde_json::json!({ "asset": asset, "sha256": sha, "sha256_file": sha_file })
    } else {
        // Dev: plain copy.
        std::fs::copy(&daemon_src, dist.join(DAEMON_CRATE))
            .map_err(|e| format!("copying daemon: {e}"))?;
        eprintln!("packaging:   daemon '{DAEMON_CRATE}' -> dist/{DAEMON_CRATE}");
        serde_json::json!({ "asset": DAEMON_CRATE })
    };

    // 5. Manifest.
    let manifest = serde_json::json!({
        "provider_abi_version": PROVIDER_ABI_VERSION,
        "profile": profile,
        "arch": std::env::consts::ARCH,
        "daemon": daemon,
        "providers": providers,
    });
    let manifest_path = dist.join("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("writing manifest: {e}"))?;
    eprintln!("packaging: wrote {}", manifest_path.display());
    Ok(())
}

/// The workspace root (packaging lives at `<root>/packaging`).
fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p
}

/// Lowercase hex SHA-256 of a file's contents.
fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Read a provider `.so`'s advertised name + provider-ABI version via its exported
/// C symbols. Reads only the zero-argument metadata symbols — it never calls
/// `ovata_provider_new`, so no model/config is required.
///
/// # Safety
/// `path` must be a provider library built with `export_provider!`; the metadata
/// symbols must match the expected signatures.
unsafe fn read_provider_meta(path: &Path) -> Result<(String, u32), String> {
    let lib = libloading::Library::new(path).map_err(|e| e.to_string())?;
    let abi: libloading::Symbol<unsafe extern "C" fn() -> u32> = lib
        .get(b"ovata_provider_abi_version\0")
        .map_err(|e| e.to_string())?;
    let name_ptr: libloading::Symbol<unsafe extern "C" fn() -> *const u8> = lib
        .get(b"ovata_provider_name_ptr\0")
        .map_err(|e| e.to_string())?;
    let name_len: libloading::Symbol<unsafe extern "C" fn() -> usize> = lib
        .get(b"ovata_provider_name_len\0")
        .map_err(|e| e.to_string())?;
    let version = abi();
    let name = {
        let p = name_ptr();
        let l = name_len();
        String::from_utf8_lossy(std::slice::from_raw_parts(p, l)).into_owned()
    };
    Ok((name, version))
}
