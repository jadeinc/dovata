//! Author: Joe Ricks
//! File: src/cli/main.rs
//! Description: The `dovata` command line, whose `hw` commands report the current
//! machine's memory and GPUs.

use std::io::{IsTerminal, Write};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use dovata::hw::{self, Gpu, Machine, Pressure, Usage};

const HELP: &str = "\
usage: dovata hw <command> [options]

Report this machine's memory and GPUs.

commands:
  info              memory capacity, page size, and GPU memory limits
  usage             memory in use right now
  watch             memory in use, refreshed in place until Ctrl-C
  target            this machine as the planner's target, as JSON

options:
  --json            print JSON instead of text (info, usage)
  --process NAME    also show the memory of running programs named NAME (usage, watch)
  --every SECONDS   seconds between refreshes, default 1 (watch)
  --cpu             describe the CPU even when a GPU is present (target)
";

/// Purpose: This enum lists the commands `dovata` can run, with their options.
#[derive(Debug, Clone, PartialEq)]
enum Cmd {
    /// Print the help text.
    Help,
    /// `dovata hw info`.
    Info { json: bool },
    /// `dovata hw usage`.
    Usage { json: bool, process: Option<String> },
    /// `dovata hw watch`.
    Watch { every: Duration, process: Option<String> },
    /// `dovata hw target`.
    Target { cpu: bool },
}

/// Purpose: This function runs the `dovata` command line.
///
/// Inputs: none; it reads the command-line arguments.
///
/// Outputs: success, 1 when reading the machine fails, or 2 for a bad command line.
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = match parse(&args) {
        Ok(cmd) => cmd,
        Err(msg) => {
            eprintln!("dovata: {msg}\n\n{HELP}");
            return ExitCode::from(2);
        }
    };
    match run(cmd) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("dovata: {msg}");
            ExitCode::FAILURE
        }
    }
}

/// Purpose: This function reads the command-line arguments into a command.
///
/// Inputs:
/// - `args`: the arguments after the program name, such as `["hw", "usage", "--json"]`.
///
/// Outputs: the [`Cmd`], or a message saying what is wrong with the arguments.
fn parse(args: &[String]) -> Result<Cmd, String> {
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    let (command, rest) = match words.as_slice() {
        [] | ["help" | "--help" | "-h", ..] | ["hw"] | ["hw", "help" | "--help" | "-h", ..] => return Ok(Cmd::Help),
        ["hw", command, rest @ ..] => (*command, rest),
        [other, ..] => return Err(format!("unknown command {other:?}; the commands are under `dovata hw`")),
    };
    let (mut json, mut cpu, mut process, mut every) = (false, false, None, Duration::from_secs(1));
    let mut i = 0;
    while i < rest.len() {
        let value = |i: usize| rest.get(i + 1).copied().ok_or_else(|| format!("{} needs a value", rest[i]));
        match rest[i] {
            "--json" => json = true,
            "--cpu" => cpu = true,
            "--process" => {
                process = Some(value(i)?.to_owned());
                i += 1;
            }
            "--every" => {
                let v = value(i)?;
                let secs: f64 = v.parse().map_err(|_| format!("--every takes a number of seconds, not {v:?}"))?;
                if !(0.1..=3600.0).contains(&secs) {
                    return Err(format!("--every must be between 0.1 and 3600 seconds, not {v}"));
                }
                every = Duration::from_secs_f64(secs);
                i += 1;
            }
            "--help" | "-h" => return Ok(Cmd::Help),
            other => return Err(format!("unknown option {other:?}")),
        }
        i += 1;
    }
    let allowed: &[&str] = match command {
        "info" => &["--json"],
        "usage" => &["--json", "--process"],
        "watch" => &["--process", "--every"],
        "target" => &["--cpu"],
        other => return Err(format!("unknown command `hw {other}`")),
    };
    if let Some(bad) = rest.iter().find(|w| w.starts_with("--") && !allowed.contains(w)) {
        return Err(format!("`hw {command}` does not take {bad}"));
    }
    Ok(match command {
        "info" => Cmd::Info { json },
        "usage" => Cmd::Usage { json, process },
        "watch" => Cmd::Watch { every, process },
        _ => Cmd::Target { cpu },
    })
}

/// Purpose: This function carries out one command and prints its result.
///
/// Inputs:
/// - `cmd`: the command to run.
///
/// Outputs: `Ok(())`, or the message to print when reading the machine fails.
fn run(cmd: Cmd) -> Result<(), String> {
    match cmd {
        Cmd::Help => print!("{HELP}"),
        Cmd::Info { json: as_json } => {
            let m = hw::machine().map_err(|e| e.to_string())?;
            if as_json {
                println!("{}", json(&m));
            } else {
                print!("{}", info_text(&m));
            }
        }
        Cmd::Usage { json: as_json, process } => {
            let u = hw::usage(process.as_deref()).map_err(|e| e.to_string())?;
            if as_json {
                println!("{}", json(&u));
            } else {
                print!("{}", usage_text(&u, process.as_deref()));
            }
        }
        Cmd::Watch { every, process } => watch(every, process.as_deref())?,
        Cmd::Target { cpu } => {
            let m = hw::machine().map_err(|e| e.to_string())?;
            println!("{}", json(&m.target(cpu)));
        }
    }
    Ok(())
}

/// Purpose: This function writes a value as indented JSON.
///
/// Inputs:
/// - `v`: a machine, a usage reading or a planner target.
///
/// Outputs: the JSON text.
fn json<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string_pretty(v).expect("hw and plan types always serialize")
}

/// Purpose: This function shows memory in use, refreshed in place, until the program is
/// stopped or its output is closed.
///
/// Inputs:
/// - `every`: the time between refreshes.
/// - `process`: a program name whose running copies to show.
///
/// Outputs: `Ok(())` once its output is closed, as when piped into `head`, or the message
/// to print when reading the machine fails.
fn watch(every: Duration, process: Option<&str>) -> Result<(), String> {
    let machine = hw::machine().map_err(|e| e.to_string())?;
    let terminal = std::io::stdout().is_terminal();
    let start = Instant::now();
    loop {
        let u = hw::usage(process).map_err(|e| e.to_string())?;
        let text = if terminal {
            // Move to the top left and clear the screen, so each reading replaces the last.
            format!(
                "\x1b[H\x1b[2Jdovata hw watch: {}, every {} s. Press Ctrl-C to stop.\n\n{}",
                machine.chip.as_deref().unwrap_or(&machine.os),
                every.as_secs_f64(),
                usage_text(&u, process)
            )
        } else {
            format!("{:>8.1}s  {}\n", start.elapsed().as_secs_f64(), usage_line(&u))
        };
        let mut out = std::io::stdout().lock();
        // Rust ignores SIGPIPE, so a closed pipe shows up here as a write error.
        if out.write_all(text.as_bytes()).and_then(|()| out.flush()).is_err() {
            return Ok(());
        }
        drop(out);
        std::thread::sleep(every);
    }
}

/// Purpose: This function formats a machine's capacity for `dovata hw info`.
///
/// Inputs:
/// - `m`: the machine.
///
/// Outputs: the text, one item per line.
fn info_text(m: &Machine) -> String {
    let mut s = format!("{}, {}\n", m.chip.as_deref().unwrap_or("unknown processor"), m.os);
    s += &format!("  memory      {}, {} pages\n", size(m.memory_bytes), size(m.page_bytes));
    if let Some(metal) = m.metal {
        s += &format!(
            "  Metal GPU   {} working set, {} largest buffer, {}\n",
            size(metal.working_set_bytes),
            size(metal.max_buffer_bytes),
            if metal.unified { "unified memory" } else { "its own memory" }
        );
    }
    for g in &m.nvidia {
        s += &format!("  NVIDIA {:<4} {}, {}\n", g.index, g.name, size(g.memory_total_bytes));
    }
    let target = m.target(false);
    s += &format!("  planner     {}\n", target_text(&target));
    s
}

/// Purpose: This function describes the planner's target in one phrase.
///
/// Inputs:
/// - `t`: the target.
///
/// Outputs: the phrase, such as `apple target, 17.8 GiB, buffers up to 13.3 GiB`.
fn target_text(t: &dovata::plan::Target) -> String {
    use dovata::plan::Target;
    match t {
        Target::Apple { unified_bytes, max_buffer_bytes, .. } => {
            format!("apple target, {}, buffers up to {}", size(*unified_bytes), size(*max_buffer_bytes))
        }
        Target::Cuda { vram_bytes } => format!("cuda target, {} of VRAM on GPU 0", size(*vram_bytes)),
        Target::Cpu { ram_bytes, .. } => format!("cpu target, {}", size(*ram_bytes)),
    }
}

/// Purpose: This function formats current memory use for `dovata hw usage` and `watch`.
///
/// Inputs:
/// - `u`: the reading.
/// - `process`: the program name that was asked about, if any.
///
/// Outputs: the text, one item per line.
fn usage_text(u: &Usage, process: Option<&str>) -> String {
    let m = &u.memory;
    let mut s = format!(
        "  memory     {} used of {}, {} available\n",
        size(m.used_bytes),
        size(m.total_bytes),
        size(m.available_bytes)
    );
    let parts: Vec<String> = [("wired", m.wired_bytes), ("compressed", m.compressed_bytes), ("cached", m.cached_bytes)]
        .into_iter()
        .filter_map(|(label, v)| Some(format!("{label} {}", size(v?))))
        .collect();
    if !parts.is_empty() {
        s += &format!("             {}\n", parts.join(", "));
    }
    s += &format!("  swap       {} used of {}\n", size(m.swap_used_bytes), size(m.swap_total_bytes));
    if let Some(p) = m.pressure {
        s += &format!("  pressure   {}\n", pressure_text(p));
    }
    for g in &u.nvidia {
        s += &format!("  NVIDIA {:<3} {}\n", g.index, gpu_text(g));
    }
    if let Some(name) = process {
        if u.processes.is_empty() {
            s += &format!("  process    no running program named {name}\n");
        }
        for p in &u.processes {
            s += &format!("  process    {} {}, {} resident\n", p.pid, p.command, size(p.rss_bytes));
        }
    }
    s
}

/// Purpose: This function formats current memory use as one line, for `watch` when its
/// output goes to a file or a pipe.
///
/// Inputs:
/// - `u`: the reading.
///
/// Outputs: the line.
fn usage_line(u: &Usage) -> String {
    let m = &u.memory;
    let mut s = format!("used {} of {}, swap {}", size(m.used_bytes), size(m.total_bytes), size(m.swap_used_bytes));
    if let Some(p) = m.pressure {
        s += &format!(", pressure {}", pressure_text(p));
    }
    for g in &u.nvidia {
        s += &format!(" | GPU {} {}", g.index, gpu_text(g));
    }
    for p in &u.processes {
        s += &format!(" | pid {} {}", p.pid, size(p.rss_bytes));
    }
    s
}

/// Purpose: This function formats one NVIDIA GPU's memory, load, temperature and clocks.
///
/// Inputs:
/// - `g`: the GPU.
///
/// Outputs: the text; values `nvidia-smi` could not read are left out.
fn gpu_text(g: &Gpu) -> String {
    let mut parts = vec![match g.memory_used_bytes {
        Some(used) => format!("{} used of {}", size(used), size(g.memory_total_bytes)),
        None => format!("{} total", size(g.memory_total_bytes)),
    }];
    parts.extend(g.load_percent.map(|v| format!("load {v}%")));
    parts.extend(g.temperature_c.map(|v| format!("{v} C")));
    parts.extend(g.sm_clock_mhz.map(|v| format!("{v} MHz shader")));
    parts.extend(g.memory_clock_mhz.map(|v| format!("{v} MHz memory")));
    parts.join(", ")
}

/// Purpose: This function formats a memory pressure reading.
///
/// Inputs:
/// - `p`: the reading.
///
/// Outputs: text such as `55% free` or `1.2% of the last 10 s stalled`.
fn pressure_text(p: Pressure) -> String {
    match p {
        Pressure::FreePercent(v) => format!("{v}% free"),
        Pressure::StallPercent(v) => format!("{v:.1}% of the last 10 s stalled waiting for memory"),
    }
}

/// Purpose: This function formats a byte count in binary units.
///
/// Inputs:
/// - `bytes`: the count.
///
/// Outputs: text such as `24.0 GiB`, `512.0 MiB` or `16 KiB`.
fn size(bytes: u64) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Purpose: This function reads a command line written as one string.
    ///
    /// Inputs:
    /// - `line`: the arguments, separated by spaces.
    ///
    /// Outputs: what [`parse`] returns for them.
    fn p(line: &str) -> Result<Cmd, String> {
        parse(&line.split_whitespace().map(str::to_owned).collect::<Vec<_>>())
    }

    #[test]
    fn commands_and_options_parse() {
        assert_eq!(p(""), Ok(Cmd::Help));
        assert_eq!(p("hw"), Ok(Cmd::Help));
        assert_eq!(p("hw info"), Ok(Cmd::Info { json: false }));
        assert_eq!(p("hw info --json"), Ok(Cmd::Info { json: true }));
        assert_eq!(p("hw usage --process qwen-chat"), Ok(Cmd::Usage { json: false, process: Some("qwen-chat".into()) }));
        assert_eq!(p("hw watch --every 0.5"), Ok(Cmd::Watch { every: Duration::from_millis(500), process: None }));
        assert_eq!(p("hw target --cpu"), Ok(Cmd::Target { cpu: true }));
    }

    #[test]
    fn bad_command_lines_are_refused() {
        for bad in ["plan", "hw nope", "hw info --cpu", "hw usage --process", "hw watch --every soon", "hw watch --every 0", "hw info --verbose"] {
            assert!(p(bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn sizes_use_binary_units() {
        assert_eq!(size(24 << 30), "24.0 GiB");
        assert_eq!(size(512 << 20), "512.0 MiB");
        assert_eq!(size(16384), "16 KiB");
        assert_eq!(size(0), "0 B");
    }

    #[test]
    fn usage_text_shows_every_reading() {
        let u = Usage {
            memory: hw::parse_vm_stat(include_str!("../hw/testdata/macos-vm_stat.txt"), 24 << 30).unwrap(),
            nvidia: hw::parse_nvidia_smi(include_str!("../hw/testdata/nvidia-smi.csv")).unwrap(),
            processes: hw::parse_ps(include_str!("../hw/testdata/macos-ps.txt"), "qwen-chat"),
        };
        let text = usage_text(&u, Some("qwen-chat"));
        assert!(text.contains("used of 24.0 GiB") && text.contains("wired") && text.contains("compressed"), "{text}");
        assert!(text.contains("NVIDIA 0") && text.contains("load 97%, 71 C"), "{text}");
        assert!(text.contains("process    5120 /Users/dev/models/qwen-chat, 2.3 GiB resident"), "{text}");
        let none = Usage { processes: vec![], ..u };
        assert!(usage_text(&none, Some("absent")).contains("no running program named absent"));
    }
}
