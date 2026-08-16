//! packaging — build the dovata engine `.so` + the native provider packages and
//! assemble a `dist/` bundle. This is the single source of truth for how a release is
//! built and staged; CI (`.github/workflows/release.yml`) is a thin wrapper that sets up
//! the environment and calls into here.
//!
//! Usage:
//!   cargo run -p packaging -- build [--release]     # dev bundle
//!   cargo run -p packaging -- release [--cuda]      # release bundle: assets + sha256
//!
//! What ships is libraries, not a binary. `dovata` is an `impl Provider` exported across
//! the C ABI and built as a cdylib, so a Jade program loads it and calls `infer`
//! in-process — there is no daemon and no socket. `echo` is the native reference
//! provider. The remote providers (anthropic/openai) are pure-Jade `.so` packages
//! published separately to jadelang.org (see the publish-providers workflow) and are not
//! part of this bundle.
//!
//! Every provider `.so` except the engine is validated by reading its exported name/ABI
//! symbols, which proves it is a real `export_provider!` plugin. The engine is exempt
//! because that check `dlopen`s the library and the engine links CUDA — see step 4.
use std::path::{Path, PathBuf};
use std::process::Command;

use ovata_infer_protocol::PROVIDER_ABI_VERSION;
use sha2::{Digest, Sha256};

/// The remote provider crates that ship as loadable `.so` packages.
const PROVIDER_CRATES: &[&str] = &[
    // The remote providers (anthropic, openai) are now pure-Jade libraries, not
    // native .so plugins, so they are not packaged here. `echo` is the only
    // remaining native reference provider.
    "ovata-provider-echo",
];
// The remote providers (anthropic, openai) are pure-Jade libraries published
// separately as native `.so` packages to jadelang.org (see the publish-providers
// workflow). They are NOT part of this OS bundle, which ships only the daemon (+ the
// native `echo` reference provider) to S3.

/// The engine crate: the local llama.cpp provider, shipped as a cdylib.
const ENGINE_CRATE: &str = "dovata";

/// Every capability the engine can be built with — each one a slice that compiles out
/// whole, Rust and llama.cpp seam together. This list is what a *release* must carry;
/// see [`Plan::capabilities`] for why a shipped artifact missing one is an error.
///
/// Kept in step with `[features]` in `src/dovata/Cargo.toml` by hand. The engine cannot
/// tell us its own list: `cfg!` inside the engine reports the build in front of it, not
/// the manifest, and the shipped library links CUDA so it cannot be dlopened here to be
/// asked. One short list, checked by the release path, beats a clever mechanism.
const CAPABILITIES: &[&str] = &["compose", "batch", "moe"];

/// How the bundle is built and how the engine library is staged.
struct Plan {
    /// Release profile (`target/release`) vs dev (`target/debug`).
    release: bool,
    /// Build the engine's `cuda` feature (GPU offload). Release/Linux only.
    cuda: bool,
    /// Capabilities to compile in. Defaults to all of [`CAPABILITIES`]; a dev build may
    /// narrow it with `--capabilities a,b` to cut compile time while working on one.
    capabilities: Vec<String>,
}

impl Plan {
    /// Refuse to publish an engine that is missing a capability.
    ///
    /// A `.so` built without `compose` refuses over-context prompts, and one without
    /// `moe` refuses to activate expert offload. Both are correct behaviours and both
    /// are catastrophic in a release: the artifact loads, answers ordinary prompts, and
    /// fails only on the requests the missing capability existed to serve. The manifest
    /// records what was built, and this stops the wrong thing being built at all.
    fn check_release_completeness(&self) -> Result<(), String> {
        if !self.release {
            return Ok(());
        }
        let missing: Vec<&str> = CAPABILITIES
            .iter()
            .copied()
            .filter(|c| !self.capabilities.iter().any(|h| h == c))
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        Err(format!(
            "refusing to build a release without every capability: missing {}. \
             A partial engine loads and serves ordinary prompts, then fails only on the \
             requests the missing capability existed to serve. Drop --capabilities, or \
             use `build` rather than `release` for a narrowed dev bundle.",
            missing.join(", ")
        ))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    // `--capabilities a,b` narrows what the engine is built with. Absent, everything.
    let capabilities = match args.iter().position(|a| a == "--capabilities") {
        Some(i) => match args.get(i + 1) {
            Some(csv) => csv.split(',').map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).collect(),
            None => {
                eprintln!("packaging: --capabilities needs a comma-separated list");
                std::process::exit(2);
            }
        },
        None => CAPABILITIES.iter().map(|s| (*s).to_owned()).collect(),
    };
    let plan = match args.first().map(String::as_str) {
        Some("build") => Plan { release: has("--release") || has("-r"), cuda: false, capabilities },
        Some("release") => Plan { release: true, cuda: has("--cuda"), capabilities },
        _ => {
            eprintln!(
                "usage: cargo run -p packaging -- <build [--release] | release [--cuda]> \
                 [--capabilities a,b]"
            );
            std::process::exit(2);
        }
    };
    if let Err(e) = plan.check_release_completeness().and_then(|()| assemble(&plan)) {
        eprintln!("packaging: error: {e}");
        std::process::exit(1);
    }
}

fn assemble(plan: &Plan) -> Result<(), String> {
    let root = workspace_root();
    let profile = if plan.release { "release" } else { "debug" };
    let target_dir = root.join("target").join(profile);
    let dist = root.join("dist");

    // 1. Build the engine + every provider cdylib in one cargo invocation.
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut cmd = Command::new(&cargo);
    cmd.arg("build").current_dir(&root);
    if plan.release {
        cmd.arg("--release");
    }
    cmd.args(["-p", ENGINE_CRATE]);
    // Capabilities are passed explicitly rather than left to `default`, so the manifest
    // below records what was actually asked for instead of what the manifest happened to
    // default to. `--no-default-features` makes the list authoritative: with defaults
    // still on, dropping a capability from the list would have no effect and the
    // recorded value would be a lie.
    let mut features: Vec<&str> = plan.capabilities.iter().map(String::as_str).collect();
    if plan.cuda {
        features.push("cuda");
    }
    cmd.arg("--no-default-features");
    if !features.is_empty() {
        cmd.args(["--features", &features.join(",")]);
    }
    for c in PROVIDER_CRATES {
        cmd.args(["-p", c]);
    }
    eprintln!(
        "packaging: building {ENGINE_CRATE} [{}]{} + {} providers ({profile})",
        plan.capabilities.join(" "),
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
        let (name, abi, protocol) = unsafe { read_provider_meta(&src) }
            .map_err(|e| format!("{crate_name}: not a valid provider .so: {e}"))?;
        if abi != PROVIDER_ABI_VERSION {
            eprintln!(
                "packaging: WARNING {crate_name} advertises ABI {abi} != host ABI {PROVIDER_ABI_VERSION}"
            );
        }

        let dest = dist.join(&file_name);
        std::fs::copy(&src, &dest).map_err(|e| format!("copying {}: {e}", src.display()))?;
        eprintln!(
            "packaging:   provider '{name}' (abi {abi}, protocol {}) -> dist/{}",
            protocol.map_or_else(|| "?".to_string(), |v| v.to_string()),
            file_name.to_string_lossy()
        );
        providers.push(serde_json::json!({
            "crate": crate_name,
            "name": name,
            "abi_version": abi,
            "protocol_version": protocol,
            "file": file_name.to_string_lossy(),
        }));
    }

    // 4. Stage the engine cdylib — the artifact that ships.
    //
    // dovata used to build a daemon binary too, served over a Unix socket. That is
    // gone: the engine is loaded in-process by the Jade program, so a library is the
    // whole product. GPU health checking, the one job that genuinely wants a resident
    // process, lives in its own daemon now.
    //
    // Deliberately NOT run through `read_provider_meta` like the other providers: that
    // dlopens the library, and this one links CUDA, so it cannot load on a builder with
    // no NVIDIA driver — which the release runner is. Validating it here would fail
    // every release. Its ABI holds by construction instead, from the same
    // `export_provider!` on the same trait impl.
    //
    // Linux-only because the engine is. Elsewhere the cdylib is an empty stub, and
    // shipping something that looks like a provider but exports nothing is worse than
    // shipping nothing.
    let engine = if cfg!(target_os = "linux") {
        let lib_src = target_dir.join(libloading::library_filename(ENGINE_CRATE));
        if !lib_src.exists() {
            return Err(format!("expected engine cdylib missing: {}", lib_src.display()));
        }
        let asset = format!("{ENGINE_CRATE}-{}.so", std::env::consts::ARCH);
        let dest = dist.join(&asset);
        std::fs::copy(&lib_src, &dest).map_err(|e| format!("staging engine library: {e}"))?;
        let sha = sha256_file(&dest)?;
        let sha_file = format!("{asset}.sha256");
        std::fs::write(dist.join(&sha_file), format!("{sha}  {asset}\n"))
            .map_err(|e| format!("writing {sha_file}: {e}"))?;
        eprintln!("packaging:   engine library -> dist/{asset} (+ .sha256)");
        serde_json::json!({
            "asset": asset,
            "sha256": sha,
            "sha256_file": sha_file,
            "provider_name": "dovata",
            // Not read from the library — it cannot be dlopened here (see above) —
            // but from the same protocol crate it was compiled against, which this
            // xtask links too. Same constants, same build.
            "abi_version": PROVIDER_ABI_VERSION,
            "protocol_version": ovata_infer_protocol::PROTOCOL_VERSION,
            // What this particular .so can do. A consumer holding the file cannot ask
            // it — dlopening a CUDA-linked library needs a driver — so the answer ships
            // beside it, the same reasoning as the model's .buildinfo sidecar.
            "capabilities": plan.capabilities,
        })
    } else {
        eprintln!("packaging:   engine library skipped (engine is Linux-only)");
        serde_json::Value::Null
    };

    // 5. Manifest. Consumers should read asset names from here rather than hardcoding
    // them — the OS install layer and the language both need to find these files.
    let manifest = serde_json::json!({
        "provider_abi_version": PROVIDER_ABI_VERSION,
        "protocol_version": ovata_infer_protocol::PROTOCOL_VERSION,
        "profile": profile,
        "arch": std::env::consts::ARCH,
        "engine": engine,
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

/// The workspace root. This crate lives at `<root>/src/packaging`, and cargo's
/// `target/` plus the `dist/` bundle both sit at `<root>`, so pop twice.
fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // <root>/src
    p.pop(); // <root>
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
unsafe fn read_provider_meta(path: &Path) -> Result<(String, u32, Option<u32>), String> {
    let lib = libloading::Library::new(path).map_err(|e| e.to_string())?;
    let abi: libloading::Symbol<unsafe extern "C" fn() -> u32> = lib
        .get(b"ovata_provider_abi_version\0")
        .map_err(|e| e.to_string())?;
    // Optional: libraries built before this symbol existed simply do not export it,
    // and that is not a reason to reject an otherwise valid provider.
    let proto: Option<libloading::Symbol<unsafe extern "C" fn() -> u32>> =
        lib.get(b"ovata_provider_protocol_version\0").ok();
    let name_ptr: libloading::Symbol<unsafe extern "C" fn() -> *const u8> = lib
        .get(b"ovata_provider_name_ptr\0")
        .map_err(|e| e.to_string())?;
    let name_len: libloading::Symbol<unsafe extern "C" fn() -> usize> = lib
        .get(b"ovata_provider_name_len\0")
        .map_err(|e| e.to_string())?;
    let version = abi();
    let protocol = proto.map(|f| f());
    let name = {
        let p = name_ptr();
        let l = name_len();
        String::from_utf8_lossy(std::slice::from_raw_parts(p, l)).into_owned()
    };
    Ok((name, version, protocol))
}
