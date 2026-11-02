//! Author: Joe Ricks
//! File: src/cli/machine.rs
//! Description: The `dovata machine` commands, which report the current machine's memory
//! and GPUs.

use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

use dovata::machine::{self, Gpu, Machine, Pressure, Usage};

use super::{json, out, size, target_text, Fail};

/// The help text for `dovata machine`.
pub(crate) const HELP: &str = "\
usage: dovata machine <command> [options]

Report this machine's memory and GPUs.

commands:
  info              memory capacity, page size, and GPU memory limits
  usage             memory in use right now
  watch             memory in use, refreshed in place until Ctrl-C
  target            the memory and limits the planner will use on this machine

options:
  --json            print JSON instead of text (info, usage, target)
  --process NAME    also show the memory of running programs named NAME (usage, watch)
  --every SECONDS   seconds between refreshes, default 1 (watch)
  --cpu             describe the CPU even when a GPU is present (target)
";

/// Purpose: This enum lists the commands `dovata` can run, with their options.
#[derive(Debug, Clone, PartialEq)]
enum Cmd {
    /// Print the help text.
    Help,
    /// `dovata machine info`.
    Info { json: bool },
    /// `dovata machine usage`.
    Usage { json: bool, process: Option<String> },
    /// `dovata machine watch`.
    Watch { every: Duration, process: Option<String> },
    /// `dovata machine target`.
    Target { json: bool, cpu: bool },
}

/// Purpose: This function runs one `dovata machine` command.
///
/// Inputs:
/// - `args`: the arguments after `machine`, such as `["usage", "--json"]`.
///
/// Outputs: `Ok(())`, or a [`Fail`] for a bad command line or a failed reading.
pub(crate) fn main(
    args: &[String],
) -> Result<(), Fail>
{
    let cmd = parse(args).map_err(|msg| Fail::Usage(format!("{msg}\n\n{HELP}")))?;
    run(cmd).map_err(Fail::Run)
}

/// Purpose: This function reads the arguments of a `dovata machine` command.
///
/// Inputs:
/// - `args`: the arguments after `machine`, such as `["usage", "--json"]`.
///
/// Outputs: the [`Cmd`], or a message saying what is wrong with the arguments.
fn parse(
    args: &[String],
) -> Result<Cmd, String>
{
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    let (command, rest) = match words.as_slice() {
        [] | ["help" | "--help" | "-h", ..] => return Ok(Cmd::Help),
        [command, rest @ ..] => (*command, rest),
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
        "target" => &["--json", "--cpu"],
        other => return Err(format!("unknown command `machine {other}`")),
    };
    if let Some(bad) = rest.iter().find(|w| w.starts_with("--") && !allowed.contains(w)) {
        return Err(format!("`machine {command}` does not take {bad}"));
    }
    Ok(match command {
        "info" => Cmd::Info { json },
        "usage" => Cmd::Usage { json, process },
        "watch" => Cmd::Watch { every, process },
        _ => Cmd::Target { json, cpu },
    })
}

/// Purpose: This function carries out one command and prints its result.
///
/// Inputs:
/// - `cmd`: the command to run.
///
/// Outputs: `Ok(())`, or the message to print when reading the machine fails.
fn run(
    cmd: Cmd,
) -> Result<(), String>
{
    match cmd {
        Cmd::Help => out(HELP),
        Cmd::Info { json: as_json } => {
            let m = machine::current().map_err(|e| e.to_string())?;
            if as_json {
                out(&format!("{}\n", json(&m)));
            } else {
                out(&info_text(&m));
            }
        }
        Cmd::Usage { json: as_json, process } => {
            let u = machine::usage(process.as_deref()).map_err(|e| e.to_string())?;
            if as_json {
                out(&format!("{}\n", json(&u)));
            } else {
                out(&usage_text(&u, process.as_deref()));
            }
        }
        Cmd::Watch { every, process } => watch(every, process.as_deref())?,
        Cmd::Target { json: as_json, cpu } => {
            let target = machine::current().map_err(|e| e.to_string())?.target(cpu);
            if as_json {
                out(&format!("{}\n", json(&target)));
            } else {
                out(&target_block(&target));
            }
        }
    }
    Ok(())
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
fn watch(
    every: Duration,
    process: Option<&str>,
) -> Result<(), String>
{
    let m = machine::current().map_err(|e| e.to_string())?;
    let terminal = std::io::stdout().is_terminal();
    let start = Instant::now();
    loop {
        let u = machine::usage(process).map_err(|e| e.to_string())?;
        let text = if terminal {
            // Move to the top left and clear the screen, so each reading replaces the last.
            format!(
                "\x1b[H\x1b[2Jdovata machine watch: {}, every {} s. Press Ctrl-C to stop.\n\n{}",
                m.chip.as_deref().unwrap_or(&m.os),
                every.as_secs_f64(),
                usage_text(&u, process)
            )
        } else {
            format!("{:>8.1}s  {}\n", start.elapsed().as_secs_f64(), usage_line(&u))
        };
        let mut stdout = std::io::stdout().lock();
        // Rust ignores SIGPIPE, so a closed pipe shows up here as a write error.
        if stdout.write_all(text.as_bytes()).and_then(|()| stdout.flush()).is_err() {
            return Ok(());
        }
        drop(stdout);
        std::thread::sleep(every);
    }
}

/// Purpose: This function formats a machine's capacity for `dovata machine info`.
///
/// Inputs:
/// - `m`: the machine.
///
/// Outputs: the text, one item per line.
fn info_text(
    m: &Machine,
) -> String
{
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

/// Purpose: This function formats the planner's target for `dovata machine target`.
///
/// Inputs:
/// - `t`: the target.
///
/// Outputs: the text, the kind of target first and then one limit per line.
fn target_block(
    t: &dovata::plan::Target,
) -> String
{
    use dovata::plan::Target;
    match t {
        Target::Apple { unified_bytes, max_buffer_bytes, page_bytes } => format!(
            "apple target\n  memory          {} the plan may use\n  largest buffer  {}\n  page size       {}\n",
            size(*unified_bytes),
            size(*max_buffer_bytes),
            size(*page_bytes)
        ),
        Target::Cuda { vram_bytes } => {
            format!("cuda target, GPU 0\n  memory          {} of VRAM the plan may use\n", size(*vram_bytes))
        }
        Target::Cpu { ram_bytes, page_bytes } => format!(
            "cpu target\n  memory          {} the plan may use\n  page size       {}\n",
            size(*ram_bytes),
            size(*page_bytes)
        ),
    }
}

/// Purpose: This function formats current memory use for `dovata machine usage` and `watch`.
///
/// Inputs:
/// - `u`: the reading.
/// - `process`: the program name that was asked about, if any.
///
/// Outputs: the text, one item per line.
fn usage_text(
    u: &Usage,
    process: Option<&str>,
) -> String
{
    let m = &u.memory;
    let mut s = format!(
        "  memory     {} used of {}, {} available\n",
        size(m.used_bytes),
        size(m.total_bytes),
        size(m.available_bytes)
    );
    let parts: Vec<String> = [("immovable", m.immovable_bytes), ("compressed", m.compressed_bytes), ("cached", m.cached_bytes)]
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
fn usage_line(
    u: &Usage,
) -> String
{
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
fn gpu_text(
    g: &Gpu,
) -> String
{
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
fn pressure_text(
    p: Pressure,
) -> String
{
    match p {
        Pressure::FreePercent(v) => format!("{v}% free"),
        Pressure::StallPercent(v) => format!("{v:.1}% of the last 10 s stalled waiting for memory"),
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
    fn p(
        line: &str,
    ) -> Result<Cmd, String>
    {
        parse(&line.split_whitespace().map(str::to_owned).collect::<Vec<_>>())
    }

    #[test]
    fn commands_and_options_parse()
    {
        assert_eq!(p(""), Ok(Cmd::Help));
        assert_eq!(p("help"), Ok(Cmd::Help));
        assert_eq!(p("info"), Ok(Cmd::Info { json: false }));
        assert_eq!(p("info --json"), Ok(Cmd::Info { json: true }));
        assert_eq!(p("usage --process qwen-chat"), Ok(Cmd::Usage { json: false, process: Some("qwen-chat".into()) }));
        assert_eq!(p("watch --every 0.5"), Ok(Cmd::Watch { every: Duration::from_millis(500), process: None }));
        assert_eq!(p("target --cpu"), Ok(Cmd::Target { json: false, cpu: true }));
        assert_eq!(p("target --json"), Ok(Cmd::Target { json: true, cpu: false }));
    }

    #[test]
    fn bad_command_lines_are_refused()
    {
        for bad in ["nope", "info --cpu", "usage --process", "watch --every soon", "watch --every 0", "info --verbose"] {
            assert!(p(bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn target_text_shows_units()
    {
        let apple = dovata::plan::Target::Apple { unified_bytes: 19_069_665_280, max_buffer_bytes: 14_302_248_960, page_bytes: 16384 };
        assert_eq!(
            target_block(&apple),
            "apple target\n  memory          17.8 GiB the plan may use\n  largest buffer  13.3 GiB\n  page size       16 KiB\n"
        );
        let cuda = dovata::plan::Target::Cuda { vram_bytes: 80 << 30 };
        assert!(target_block(&cuda).contains("80.0 GiB of VRAM"));
    }

    #[test]
    fn usage_text_shows_every_reading()
    {
        let u = Usage {
            memory: machine::parse_vm_stat(include_str!("../machine/testdata/macos-vm_stat.txt"), 24 << 30).unwrap(),
            nvidia: machine::parse_nvidia_smi(include_str!("../machine/testdata/nvidia-smi.csv")).unwrap(),
            processes: machine::parse_ps(include_str!("../machine/testdata/macos-ps.txt"), "qwen-chat"),
        };
        let text = usage_text(&u, Some("qwen-chat"));
        assert!(text.contains("used of 24.0 GiB") && text.contains("immovable") && text.contains("compressed"), "{text}");
        assert!(text.contains("NVIDIA 0") && text.contains("load 97%, 71 C"), "{text}");
        assert!(text.contains("process    5120 /Users/dev/models/qwen-chat, 2.3 GiB resident"), "{text}");
        let none = Usage { processes: vec![], ..u };
        assert!(usage_text(&none, Some("absent")).contains("no running program named absent"));
    }
}
