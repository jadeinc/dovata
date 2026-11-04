//! Author: Joe Ricks
//! File: src/host/process_test.rs
//! Description: Starts the real `dovata` program the way a Jade program does, adds a
//! tiny model to it, maps its memory into this test, checks the weights, and stops it,
//! through both the library and the command line.

use std::path::Path;
use std::process::Command;

use dovata::gguf::write::Writer;
use dovata::gguf::Value;
use dovata::host::{self, AddOptions, HostError, StartOptions, State};
use dovata::layout::BindOptions;
use dovata::load::KvType;

#[path = "fixture.rs"]
mod fixture;

/// The `dovata` program cargo built for this test.
const DOVATA: &str = env!("CARGO_BIN_EXE_dovata");

/// Purpose: This function runs the `dovata` program with the test's run directory.
///
/// Inputs:
/// - `run_dir`: the run directory.
/// - `args`: the arguments.
///
/// Outputs: the exit status's success, standard output and standard error.
fn dovata(
    run_dir: &Path,
    args: &[&str],
) -> (bool, String, String)
{
    let o = Command::new(DOVATA).args(args).env("DOVATA_RUN_DIR", run_dir).output().unwrap();
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

#[test]
fn a_program_starts_dovata_adds_a_model_maps_it_and_kills_dovata()
{
    let dir = fixture::TempDir::new();
    // The library reads the run directory from the environment, as a Jade program's
    // would. It is set before any thread starts.
    std::env::set_var("DOVATA_RUN_DIR", &dir.path);
    let model = dir.path.join("host-fixture.gguf");
    fixture::write_llama(&model);
    let step = BindOptions { n_ctx: 64, n_tokens: 8, n_outputs: 1, kv_k: KvType::F16, kv_v: KvType::F16 };
    let opts = AddOptions { step, ..AddOptions::default() };

    // Nothing runs yet.
    assert_eq!(host::connect(), Err(HostError::NotRunning));
    let (ok, _, err) = dovata(&dir.path, &["add", model.to_str().unwrap()]);
    assert!(!ok && err.contains("dovata is not running; start it with `dovata start`"), "{err}");

    // Start from code, the way a Jade program does, then add a model: a local file here,
    // a Hugging Face link the same way.
    let p = host::start(&StartOptions { program: Some(DOVATA.into()) }).unwrap();
    let s = p.status().unwrap();
    assert_ne!(s.pid, std::process::id(), "dovata is a process of its own");
    assert!(s.models.is_empty());
    assert!(matches!(host::start(&StartOptions { program: Some(DOVATA.into()) }), Err(HostError::AlreadyRunning(pid)) if pid == s.pid));
    let m = p.add(model.to_str().unwrap(), &opts).unwrap();
    assert_eq!((m.name.as_str(), m.state), ("host-fixture", State::Ready));

    // Map it, and read one weight where the plan says it is.
    let mem = p.map("host-fixture", &m.table_hash).unwrap();
    let plan = p.plan("host-fixture").unwrap();
    let e = plan["entries"].as_array().unwrap().iter().find(|e| e["tensor"] == "blk.1.ffn_down.weight").unwrap();
    let at = |k: &str| e[k].as_u64().unwrap() as usize;
    let got = mem.region(at("region") as u16).unwrap().read(at("offset"), at("bytes")).unwrap();
    let file = std::fs::read(&model).unwrap();
    let off = e["source"]["file_offset"].as_u64().unwrap() as usize;
    assert_eq!(got, &file[off..off + at("bytes")]);

    // The command line sees the same process: a second model, the status, and
    // `dovata machine usage`.
    let (ok, text, err) = dovata(&dir.path, &["start"]);
    assert!(ok && text.starts_with(&format!("dovata is already running, pid {}, with 1 model.", s.pid)), "{text}{err}");
    let (ok, text, err) = dovata(&dir.path, &["add", model.to_str().unwrap(), "--name", "second", "--ctx", "64", "--tokens", "8"]);
    assert!(ok && text.starts_with("second, added ") && text.contains("state      ready"), "{text}{err}");
    let (ok, text, err) = dovata(&dir.path, &["status"]);
    assert!(ok, "{err}");
    assert!(text.starts_with(&format!("dovata {}, pid {}", env!("CARGO_PKG_VERSION"), s.pid)) && text.contains(", 2 models, "), "{text}");
    assert!(text.contains(&format!("mapped by  pid {}, running", std::process::id())), "{text}");
    let (ok, text, err) = dovata(&dir.path, &["status", "host-fixture", "--json"]);
    assert!(ok, "{err}");
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["table_hash"].as_str(), Some(m.table_hash.as_str()));
    let (ok, text, err) = dovata(&dir.path, &["machine", "usage"]);
    assert!(ok, "{err}");
    assert!(text.contains(&format!("  dovata     pid {}, 2 models", s.pid)) && text.contains("  model      second, "), "{text}");

    // Take one model out from the command line; the other stays.
    let (ok, text, err) = dovata(&dir.path, &["remove", "second"]);
    assert!(ok && text == "second removed\n", "{text}{err}");
    let (ok, _, err) = dovata(&dir.path, &["remove", "second"]);
    assert!(!ok && err.contains("no model named second"), "{err}");
    assert_eq!(p.status().unwrap().models.len(), 1);

    // A model the process cannot plan is refused with the process's own reason.
    let bad = dir.path.join("bad.toml");
    std::fs::write(&bad, b"not a layout [").unwrap();
    let (ok, _, err) = dovata(&dir.path, &["add", model.to_str().unwrap(), "--name", "bad", "--layout", bad.to_str().unwrap()]);
    assert!(!ok && err.contains("bad.toml"), "{err}");

    // Stop it from code. Its socket goes, and the mapping stays readable.
    p.kill().unwrap();
    assert_eq!(host::connect(), Err(HostError::NotRunning));
    assert_eq!(mem.region(at("region") as u16).unwrap().read(at("offset"), 4).unwrap(), &file[off..off + 4]);
    let (ok, text, _) = dovata(&dir.path, &["status"]);
    assert!(ok && text.starts_with("dovata is not running"), "{text}");

    // Start and stop from the command line.
    let (ok, text, err) = dovata(&dir.path, &["start"]);
    assert!(ok && text.starts_with("dovata is running, pid ") && text.contains("dovata add <model>"), "{text}{err}");
    let (ok, text, err) = dovata(&dir.path, &["add", model.to_str().unwrap(), "--ctx", "64", "--tokens", "8"]);
    assert!(ok && text.starts_with("host-fixture, added "), "{text}{err}");
    let (ok, text, err) = dovata(&dir.path, &["kill"]);
    assert!(ok && text == "dovata stopped\n", "{text}{err}");
    assert_eq!(host::connect(), Err(HostError::NotRunning));
    let (ok, _, err) = dovata(&dir.path, &["kill"]);
    assert!(!ok && err.contains("not running"), "{err}");
}
