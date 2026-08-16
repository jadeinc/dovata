//! `gguf-inspect` — print what llama.cpp and the daemon will see in a `.gguf`.
//!
//! Answers the three questions you have when someone hands you a new model file:
//! will llama.cpp recognise the architecture, does the file carry the hyperparameters
//! that architecture needs, and will dovata's profile registry match it. All three are
//! answered from the header alone, so this runs on a laptop against a file you have not
//! finished downloading.
//!
//! ```sh
//! cargo run -p gguf --bin gguf-inspect -- model.gguf
//! cargo run -p gguf --bin gguf-inspect -- model.gguf --all
//! cargo run -p gguf --bin gguf-inspect -- model.gguf --profile-stub > new-model.toml
//! ```

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::ExitCode;

use gguf::Gguf;

/// How much of the file to read. A GGUF puts all metadata before the tensor data, so
/// this is generous for any published model — and it is a cap, not a target, which is
/// what keeps `gguf-inspect` from pulling 18 GB into RAM to print forty lines.
const HEADER_BUDGET: u64 = 64 * 1024 * 1024;

/// The model names dovata ships a profile for. Kept here rather than imported so this
/// tool stays buildable on its own; the daemon's registry is the real authority.
const KNOWN_PROFILE_GLOBS: &[(&str, &str)] =
    &[("qwen3-coder-30b", "Qwen3-Coder*"), ("qwen3-coder-30b", "qwen3-coder*")];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| args.iter().any(|a| a == f);
    let Some(path) = args.iter().find(|a| !a.starts_with("--")) else {
        eprintln!(
            "usage: gguf-inspect <model.gguf> [--all] [--profile-stub]\n\
             \n\
               --all           dump every metadata key, not just the interesting ones\n\
               --profile-stub  emit a starter dovata profile TOML for this model"
        );
        return ExitCode::from(2);
    };

    let bytes = match read_head(Path::new(path)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("gguf-inspect: reading {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let model = match Gguf::parse(&bytes) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("gguf-inspect: {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    if flag("--profile-stub") {
        print_profile_stub(&model);
        return ExitCode::SUCCESS;
    }

    print_report(path, &model, flag("--all"));
    if model.report().blocking().is_empty() { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

/// Read at most [`HEADER_BUDGET`] bytes. Reading the whole file would work and would
/// also be a way to fill a laptop's RAM by accident.
fn read_head(path: &Path) -> std::io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut buf = Vec::new();
    file.take(HEADER_BUDGET).read_to_end(&mut buf)?;
    Ok(buf)
}

fn print_report(path: &str, g: &Gguf, all: bool) {
    let report = g.report();

    println!("{path}");
    println!("  gguf version    {}", g.version);
    println!("  architecture    {}", report.architecture.as_deref().unwrap_or("(missing)"));
    println!("  name            {}", report.name.as_deref().unwrap_or("(missing)"));
    println!("  tensors         {}{}", g.tensor_count, if g.truncated { " (index truncated — header-only read)" } else { "" });
    if !g.quant_mix().is_empty() {
        let mix: Vec<String> = g.quant_mix().iter().take(6).map(|(n, c)| format!("{n}×{c}")).collect();
        println!("  quant mix       {}", mix.join(", "));
    }
    if g.is_moe() {
        println!("  mixture of experts, so the MoE hparams below are load-bearing");
    }

    if all {
        println!("\nall metadata ({} keys, file order)", g.metadata.len());
        for (k, v) in &g.metadata {
            println!("  {k:<48} {v}");
        }
        // The chat template is the one value worth seeing unwrapped: it is where a new
        // model's tool-call delimiters are actually written down.
        if let Some(t) = g.string("tokenizer.chat_template") {
            println!("\ntokenizer.chat_template, verbatim");
            for line in t.lines() {
                println!("  {line}");
            }
        }
    } else {
        println!("\nhyperparameters (the architecture's own namespace)");
        let params = g.arch_params();
        if params.is_empty() {
            println!("  (none — without these the loader cannot build the model)");
        }
        for (k, v) in params {
            println!("  {k:<48} {v}");
        }
    }

    println!("\nwhat llama.cpp reads");
    for c in &report.checks {
        let mark = match (&c.found, c.required) {
            (Some(_), _) => "ok  ",
            (None, true) => "MISS",
            (None, false) => "--  ",
        };
        let shown = c.found.as_deref().unwrap_or("(absent)");
        println!("  {mark} {:<44} {}", c.key, elide(shown, 60));
    }

    println!("\ndovata profile selection");
    match report.name.as_deref() {
        Some(name) => {
            let hit = KNOWN_PROFILE_GLOBS.iter().find(|(_, pat)| glob_match(pat, name));
            match hit {
                Some((profile, pat)) => {
                    println!("  general.name = {name:?} matches {pat:?} → profiles/{profile}.toml");
                }
                None => {
                    println!("  general.name = {name:?} matches no shipped profile.");
                    println!("  The daemon will fall back to plain ChatML with no tool-call");
                    println!("  delimiters. Generation works; tool calling does not.");
                    println!("  Fix: re-run with --profile-stub and fill in the TODOs.");
                }
            }
        }
        None => println!("  general.name is absent, so no profile can match. See --profile-stub."),
    }

    let blocking = report.blocking();
    if blocking.is_empty() {
        println!("\nNothing obviously missing. This is not a promise it loads — the only");
        println!("authoritative check is loading it with the pinned llama.cpp.");
    } else {
        println!("\n{} required key(s) missing — this file will not load:", blocking.len());
        for c in blocking {
            println!("  {} — {}", c.key, c.note);
        }
    }
}

/// Emit a starter profile TOML with everything the file can answer already filled in,
/// and the model-specific tokens left as explicit TODOs. Those tokens cannot be read
/// out of a GGUF — they are a property of how the model was trained — so the honest
/// move is to point at the chat template and make the human copy them across.
fn print_profile_stub(g: &Gguf) {
    let arch = g.architecture().unwrap_or("unknown");
    let name = g.name().unwrap_or("UNKNOWN-MODEL");
    let ctx = g.u64(&format!("{arch}.context_length"));

    println!("# Model profile: {name}");
    println!("#");
    println!("# Generated by `gguf-inspect --profile-stub`. The filled-in values were read");
    println!("# from the GGUF header; every TODO below has to be answered by hand, because");
    println!("# a model's tool-call tokens are a training convention and are not recorded in");
    println!("# the file. Read `tokenizer.chat_template` (gguf-inspect --all) to find them.");
    println!("#");
    println!("# Install: save as src/model/profiles/<model>.toml, add it to BUILTIN_PROFILES");
    println!("# in src/model/mod.rs, then `cargo test -p dovata` to validate it.");
    println!();
    println!("schema_version = 1");
    println!("model = {name:?}");
    println!();
    println!("# Matched against `general.name` from the GGUF, which is exactly the string");
    println!("# above. `*` is the only wildcard. Keep one glob loose enough to survive a");
    println!("# requant by a different publisher.");
    println!("match = [{:?}, {:?}]", name, format!("{}*", name.split('-').next().unwrap_or(name)));
    println!();
    println!("[tool_call]");
    println!("# TODO the delimiters this model emits around a tool invocation.");
    println!("open = \"<tool_call>\"      # TODO verify against tokenizer.chat_template");
    println!("close = \"</tool_call>\"    # TODO verify");
    println!("name_field = \"name\"       # TODO the JSON field naming the tool");
    println!();
    println!("[tool_result]");
    println!("# TODO how a tool result is rendered back on the next turn. Getting this wrong");
    println!("# does not error — the model just stops binding results to their calls.");
    println!("role = \"tool\"");
    println!("wrap_prefix = \"\"");
    println!("wrap_suffix = \"\"");
    println!();
    println!("[limits]");
    match ctx {
        Some(n) => println!("context_window = {n}    # from {arch}.context_length"),
        None => println!("# context_window = ?    # TODO: {arch}.context_length was absent"),
    }
    println!("# rlm_max_context = ?    # measured, not read from the file; omit to default");

    if let Some(t) = g.string("tokenizer.chat_template") {
        println!();
        println!("# ---- tokenizer.chat_template, verbatim, for working out the TODOs ----");
        for line in t.lines() {
            println!("# {line}");
        }
    } else {
        println!();
        println!("# This GGUF carries no tokenizer.chat_template, so the delimiters have to");
        println!("# come from the model card.");
    }
}

/// `*`-glob match, same rule as the daemon's profile registry.
fn glob_match(pattern: &str, text: &str) -> bool {
    let (p, t) = (pattern.as_bytes(), text.as_bytes());
    let (mut pi, mut ti) = (0, 0);
    let (mut star, mut mark) = (usize::MAX, 0);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star = pi;
            mark = ti;
            pi += 1;
        } else if star != usize::MAX {
            pi = star + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

fn elide(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    format!("{}…", s.chars().take(max).collect::<String>())
}

#[cfg(test)]
mod tests {
    use super::glob_match;

    #[test]
    fn globs_match_the_way_the_daemon_does() {
        assert!(glob_match("Qwen3-Coder*", "Qwen3-Coder-30B-A3B-Instruct"));
        assert!(!glob_match("Qwen3-Coder*", "Qwen3.5-Coder-30B"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("Qwen3*Instruct", "Qwen3-Coder-30B-Instruct"));
    }
}
