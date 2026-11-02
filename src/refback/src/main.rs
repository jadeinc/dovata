//! Author: Joe Ricks
//! File: src/refback/src/main.rs
//! Description: The `refback` dev command line: plans a model the way `dovata plan` does
//! and checks the plan on real memory, one line per check.

use std::path::Path;
use std::process::ExitCode;

use dovata::layout::{self, BindOptions};
use dovata::load::{KvType, Model};
use dovata::target::Target;

/// The help text.
const HELP: &str = "\
usage: refback <model.gguf> [--target cpu|apple]

Plan a model with dovata's built-in layout for this machine, set up the plan's memory
for real, and check every tensor: the file's identity, each region's base, each entry's
alignment, each weight's bytes and a matvec over them, and the step's memory sharing.
Prints one pass or fail line per check, and exits with 1 when any check fails.

  --target cpu     host memory and a CPU matvec (the default)
  --target apple   shared and no-copy Metal buffers and a Metal matvec; needs a build
                   with --features metal

The step is dovata plan's default: 512 tokens, 1 with logits, a 4096-token f16 KV cache.
";

/// Purpose: This function runs the `refback` command line.
///
/// Inputs: none; it reads the command-line arguments.
///
/// Outputs: success when every check passes, 1 when a check fails or refback refuses to
/// start, and 2 for a bad command line.
fn main() -> ExitCode
{
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| matches!(a.as_str(), "help" | "--help" | "-h")) {
        print!("{HELP}");
        return ExitCode::SUCCESS;
    }
    let (model, apple) = match parse(&args) {
        Ok(v) => v,
        Err(m) => {
            eprintln!("refback: {m}\n\n{HELP}");
            return ExitCode::from(2);
        }
    };
    match run(Path::new(&model), apple) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(m) => {
            eprintln!("refback: {m}");
            ExitCode::FAILURE
        }
    }
}

/// Purpose: This function reads the command line.
///
/// Inputs:
/// - `args`: the arguments, such as `["model.gguf", "--target", "apple"]`.
///
/// Outputs: the model's path and whether the target is Apple, or a message.
fn parse(
    args: &[String],
) -> Result<(String, bool), String>
{
    let (mut model, mut apple) = (None, false);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => {
                apple = match args.get(i + 1).map(String::as_str) {
                    Some("cpu") => false,
                    Some("apple") => true,
                    Some(other) => return Err(format!("--target takes cpu or apple, not {other:?}")),
                    None => return Err("--target needs cpu or apple".into()),
                };
                i += 1;
            }
            f if f.starts_with("--") => return Err(format!("unknown option {f:?}")),
            m if model.is_none() => model = Some(m.to_owned()),
            m => return Err(format!("one model at a time; {m:?} is a second one")),
        }
        i += 1;
    }
    Ok((model.ok_or("name the model file")?, apple))
}

/// Purpose: This function plans the model, runs every check and prints the results.
///
/// Inputs:
/// - `path`: the model file, or its first part.
/// - `apple`: whether to plan for this Mac's Metal GPU rather than its CPU.
///
/// Outputs: whether every check passed, or a message when the model cannot be planned or
/// refback refuses to start.
fn run(
    path: &Path,
    apple: bool,
) -> Result<bool, String>
{
    let model = Model::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let layout = layout::for_model(&model, None).map_err(|e| e.to_string())?;
    let machine = dovata::machine::current().map_err(|e| e.to_string())?;
    let target = match (apple, machine.target(!apple)) {
        (true, t @ Target::Apple { .. }) => t,
        (true, _) => return Err("this machine has no Metal GPU".into()),
        (false, t) => t,
    };
    let opts = BindOptions { n_ctx: 4096, n_tokens: 512, n_outputs: 1, kv_k: KvType::F16, kv_v: KvType::F16 };
    let plan = dovata::memplan::plan(&model, &layout, &opts, &target).map_err(|e| e.to_string())?;
    println!(
        "refback: {} ({}), {} regions and {} entries planned for {:?}, table hash {}",
        path.display(),
        model.architecture(),
        plan.regions.len(),
        plan.entries.len(),
        plan.target,
        plan.table_hash
    );
    let report = refback::run(&model.paths, &plan).map_err(|e| e.to_string())?;
    for o in &report.outcomes {
        println!("{o}");
    }
    let failed = report.failures().len();
    println!("refback: {} checks, {} passed, {failed} failed", report.outcomes.len(), report.outcomes.len() - failed);
    Ok(failed == 0)
}
