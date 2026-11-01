//! Author: Joe Ricks
//! File: src/cli/main.rs
//! Description: The `dovata` command line: `dovata machine` reports this machine, and
//! `dovata plan` places a model's tensors in memory.

mod machine;
mod plan;

use std::io::Write;
use std::process::ExitCode;

/// The help text for `dovata` with no command.
const HELP: &str = "\
usage: dovata <command> ...

commands:
  machine   report this machine's memory and GPUs; see `dovata machine help`
  plan      place a model's tensors in memory for one machine; see `dovata plan --help`
";

/// Purpose: This enum lists the ways a `dovata` command can fail.
pub(crate) enum Fail {
    /// The command line is wrong or incomplete. The program exits with 2.
    Usage(String),
    /// The command could not do its work. The program exits with 1.
    Run(String),
}

/// Purpose: This function runs the `dovata` command line.
///
/// Inputs: none; it reads the command-line arguments.
///
/// Outputs: success, 1 when a command fails, or 2 for a bad command line.
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        None | Some("help" | "--help" | "-h") => {
            out(HELP);
            Ok(())
        }
        Some("machine") => machine::main(&args[1..]),
        Some("plan") => plan::main(&args[1..]),
        Some(other) => Err(Fail::Usage(format!("unknown command {other:?}\n\n{HELP}"))),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(Fail::Usage(msg)) => {
            eprintln!("dovata: {msg}");
            ExitCode::from(2)
        }
        Err(Fail::Run(msg)) => {
            eprintln!("dovata: {msg}");
            ExitCode::FAILURE
        }
    }
}

/// Purpose: This function writes text to standard output, and stops quietly when the
/// output has been closed, as when it is piped into `head`.
///
/// Inputs:
/// - `text`: the text.
///
/// Outputs: none.
pub(crate) fn out(text: &str) {
    // Rust ignores SIGPIPE, so a closed pipe shows up here as a write error. The reader
    // has all it wanted, so there is nothing to report.
    let mut stdout = std::io::stdout().lock();
    let _ = stdout.write_all(text.as_bytes()).and_then(|()| stdout.flush());
}

/// Purpose: This function writes a value as indented JSON.
///
/// Inputs:
/// - `v`: a machine, a usage reading, a planner target or a plan.
///
/// Outputs: the JSON text.
pub(crate) fn json<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string_pretty(v).expect("machine and plan types always serialize")
}

/// Purpose: This function formats a byte count in binary units.
///
/// Inputs:
/// - `bytes`: the count.
///
/// Outputs: text such as `24.0 GiB`, `512.0 MiB` or `16 KiB`.
pub(crate) fn size(bytes: u64) -> String {
    const KIB: u64 = 1 << 10;
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    match bytes {
        b if b >= GIB => format!("{:.1} GiB", b as f64 / GIB as f64),
        b if b >= MIB => format!("{:.1} MiB", b as f64 / MIB as f64),
        b if b >= KIB && b % KIB == 0 => format!("{} KiB", b / KIB),
        b => format!("{b} B"),
    }
}

/// Purpose: This function reads a byte count written with a binary unit, such as `64G`.
///
/// Inputs:
/// - `s`: the text: a number, then optionally `K`, `M`, `G` or `T`, each followed by
///   nothing, `B` or `iB`. Every unit is a power of 1024, so `1G` is 1,073,741,824
///   bytes.
///
/// Outputs: the byte count, or a message saying what is wrong with `s`.
pub(crate) fn parse_size(s: &str) -> Result<u64, String> {
    let bad = || format!("{s:?} is not a size; write one such as 64G, 512M or 16K");
    let t = s.trim();
    let digits = t.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(t.len());
    let (number, unit) = t.split_at(digits);
    let shift = match unit.to_ascii_uppercase().as_str() {
        "" | "B" => 0,
        "K" | "KB" | "KIB" => 10,
        "M" | "MB" | "MIB" => 20,
        "G" | "GB" | "GIB" => 30,
        "T" | "TB" | "TIB" => 40,
        _ => return Err(bad()),
    };
    let value: f64 = number.parse().map_err(|_| bad())?;
    let bytes = value * (1u64 << shift) as f64;
    if !bytes.is_finite() || bytes < 1.0 || bytes > u64::MAX as f64 || bytes.fract() != 0.0 {
        return Err(bad());
    }
    Ok(bytes as u64)
}

/// Purpose: This function describes the planner's target in one phrase.
///
/// Inputs:
/// - `t`: the target.
///
/// Outputs: the phrase, such as `apple target, 17.8 GiB, buffers up to 13.3 GiB`.
pub(crate) fn target_text(t: &dovata::plan::Target) -> String {
    use dovata::plan::Target;
    match t {
        Target::Apple { unified_bytes, max_buffer_bytes, .. } => {
            format!("apple target, {}, buffers up to {}", size(*unified_bytes), size(*max_buffer_bytes))
        }
        Target::Cuda { vram_bytes } => format!("cuda target, {} of VRAM on GPU 0", size(*vram_bytes)),
        Target::Cpu { ram_bytes, .. } => format!("cpu target, {}", size(*ram_bytes)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_use_binary_units() {
        assert_eq!(size(24 << 30), "24.0 GiB");
        assert_eq!(size(512 << 20), "512.0 MiB");
        assert_eq!(size(16384), "16 KiB");
        assert_eq!(size(0), "0 B");
    }

    #[test]
    fn sizes_parse_with_binary_units() {
        assert_eq!(parse_size("64G"), Ok(64 << 30));
        assert_eq!(parse_size("64GiB"), Ok(64 << 30));
        assert_eq!(parse_size("16k"), Ok(16384));
        assert_eq!(parse_size("1.5G"), Ok(3 << 29));
        assert_eq!(parse_size("4096"), Ok(4096));
        for bad in ["", "G", "64X", "-1G", "0", "0.3"] {
            assert!(parse_size(bad).is_err(), "{bad:?} should be refused");
        }
    }
}
