//! Author: Joe Ricks
//! File: src/gguf/main.rs
//! Description: Prints what is in a GGUF file's header, as the `gguf-inspect` command.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::ExitCode;

use dovata::gguf::Gguf;

/// How much of the file to read. A GGUF puts all metadata before the tensor data, so
/// this is generous for any published model — and it is a cap, not a target, which is
/// what keeps `gguf-inspect` from pulling 18 GB into RAM to print forty lines.
const HEADER_BUDGET: u64 = 64 * 1024 * 1024;

/// Purpose: This function runs `gguf-inspect` on the file named on the command line.
///
/// Inputs: none.
///
/// Outputs: exit status 0 when nothing required is missing, 1 when something is or the
/// file is unreadable, and 2 with no path.
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| args.iter().any(|a| a == f);
    let Some(path) = args.iter().find(|a| !a.starts_with("--")) else {
        eprintln!(
            "usage: gguf-inspect <model.gguf> [--all]\n\
             \n\
               --all  dump every metadata key, not just the interesting ones"
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

    print_report(path, &model, flag("--all"));
    if model.report().blocking().is_empty() { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

/// Purpose: This function reads at most [`HEADER_BUDGET`] bytes from the start of a file.
///
/// Inputs:
/// - `path`: the file.
///
/// Outputs: the bytes, or an I/O error.
fn read_head(path: &Path) -> std::io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut buf = Vec::new();
    file.take(HEADER_BUDGET).read_to_end(&mut buf)?;
    Ok(buf)
}

/// Purpose: This function prints the `gguf-inspect` report for one GGUF header.
///
/// Inputs:
/// - `path`: the file's path, printed as the title.
/// - `g`: the parsed header.
/// - `all`: whether to print every metadata key.
///
/// Outputs: none.
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
    if let Some(ft) = g.file_type() {
        println!("  file type       {ft} (general.file_type, a llama_ftype)");
    }
    if let Some(s) = g.split() {
        println!("  split           part {} of {}, {} tensors across all parts", s.no + 1, s.count, s.tensors_count);
    }
    match g.data_offset {
        Some(off) => println!("  tensor data at  byte {off} (alignment {})", g.alignment),
        None => println!("  tensor data at  unknown until the whole index is read"),
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

    println!("\ndovata layout");
    match report.architecture.as_deref() {
        Some(arch) if dovata::layout::builtin(arch).is_some() => {
            println!("  {arch} has a built-in layout, so dovata can plan this file.");
        }
        Some(arch) => {
            println!("  {arch} has no built-in layout. dovata plans it only with a layout file");
            println!("  you supply. Built in: {}.", dovata::layout::SUPPORTED.join(", "));
        }
        None => println!("  no architecture, so no layout applies."),
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

/// Purpose: This function shortens a string for display, adding an ellipsis when it cuts.
///
/// Inputs:
/// - `s`: the text.
/// - `max`: the most characters to keep.
///
/// Outputs: the shortened text.
fn elide(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    format!("{}…", s.chars().take(max).collect::<String>())
}
