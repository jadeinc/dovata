//! Author: Joe Ricks
//! File: src/host/process_test.rs
//! Description: Starts the real `dovata` program on a tiny model the way a Jade program
//! does, maps its memory into this test, checks the weights, and stops it, through both
//! the library and the command line.

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use dovata::gguf::write::Writer;
use dovata::gguf::Value;
use dovata::host::{self, HostError, StartOptions, State};
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
fn a_program_starts_maps_and_kills_a_dovata_process()
{
    let dir = fixture::TempDir::new();
    // The library reads the run directory from the environment, as a Jade program's
    // would. It is set before any thread starts.
    std::env::set_var("DOVATA_RUN_DIR", &dir.path);
    let model = dir.path.join("host-fixture.gguf");
    fixture::write_llama(&model);
    let step = BindOptions { n_ctx: 64, n_tokens: 8, n_outputs: 1, kv_k: KvType::F16, kv_v: KvType::F16 };

    // Start from code, the way a Jade program does: a local file here, a Hugging Face
    // link the same way.
    let opts = StartOptions { step, program: Some(DOVATA.into()), ..StartOptions::default() };
    let h = host::start(model.to_str().unwrap(), &opts).unwrap();
    assert_eq!(h.name(), "host-fixture");
    let s = h.status().unwrap();
    assert_eq!(s.state, State::Ready);
    assert_ne!(s.pid, std::process::id(), "the model is held by a process of its own");
    assert!(matches!(host::start_parts(std::slice::from_ref(&model), &opts), Err(HostError::AlreadyRunning(n)) if n == "host-fixture"));

    // Map it, and read one weight where the plan says it is.
    let m = h.map(&s.table_hash).unwrap();
    let plan = h.plan().unwrap();
    let e = plan["entries"].as_array().unwrap().iter().find(|e| e["tensor"] == "blk.1.ffn_down.weight").unwrap();
    let at = |k: &str| e[k].as_u64().unwrap() as usize;
    let got = m.region(at("region") as u16).unwrap().read(at("offset"), at("bytes")).unwrap();
    let file = std::fs::read(&model).unwrap();
    let off = e["source"]["file_offset"].as_u64().unwrap() as usize;
    assert_eq!(got, &file[off..off + at("bytes")]);

    // The command line sees the same process, and so does `dovata machine usage`.
    let (ok, text, err) = dovata(&dir.path, &["status"]);
    assert!(ok, "{err}");
    assert!(text.starts_with(&format!("host-fixture, pid {}", s.pid)) && text.contains("state      ready"), "{text}");
    assert!(text.contains(&format!("mapped by  pid {}, running", std::process::id())), "{text}");
    let (ok, text, err) = dovata(&dir.path, &["status", "host-fixture", "--json"]);
    assert!(ok, "{err}");
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v[0]["table_hash"].as_str(), Some(s.table_hash.as_str()));
    let (ok, text, err) = dovata(&dir.path, &["machine", "usage"]);
    assert!(ok, "{err}");
    assert!(text.contains(&format!("model      host-fixture, pid {}", s.pid)), "{text}");

    // Stop it from code. Its socket goes, and the mapping stays readable.
    h.kill().unwrap();
    assert!(matches!(host::attach("host-fixture"), Err(HostError::NotRunning(_))));
    assert_eq!(m.region(at("region") as u16).unwrap().read(at("offset"), 4).unwrap(), &file[off..off + 4]);
    let (ok, text, _) = dovata(&dir.path, &["status"]);
    assert!(ok && text.starts_with("No dovata process is running"), "{text}");

    // Start and stop from the command line.
    let (ok, text, err) = dovata(&dir.path, &["start", model.to_str().unwrap(), "--name", "cli", "--ctx", "64", "--tokens", "8"]);
    assert!(ok, "{err}");
    assert!(text.starts_with("cli, pid ") && text.contains("Stop it with `dovata kill cli`"), "{text}");
    assert!(host::attach("cli").unwrap().status().unwrap().state == State::Ready);
    let (ok, text, err) = dovata(&dir.path, &["kill"]);
    assert!(ok && text == "cli stopped\n", "{text}{err}");
    assert!(host::running().unwrap().is_empty());

    // A model the process cannot plan stops the start with the process's own reason.
    let bad = dir.path.join("bad.toml");
    std::fs::write(&bad, b"not a layout [").unwrap();
    let (ok, _, err) = dovata(&dir.path, &["start", model.to_str().unwrap(), "--name", "bad", "--layout", bad.to_str().unwrap()]);
    assert!(!ok && err.contains("did not start") && err.contains("bad.toml"), "{err}");
    std::thread::sleep(Duration::from_millis(50));
    assert!(host::running().unwrap().is_empty());
}
