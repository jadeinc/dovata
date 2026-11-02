//! Author: Joe Ricks
//! File: src/cli/machine.rs
//! Description: The `dovata machine` commands, which report the current machine's memory
//! and GPUs.

use std::io::{IsTerminal, Read, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
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
  watch             memory in use on a full screen, redrawn in place until you press q
  target            the memory and limits the planner will use on this machine

options:
  --json            print JSON instead of text (info, usage, target)
  --process NAME    also show the memory of running programs named NAME (usage, watch)
  --every SECONDS   seconds between refreshes, default 1 (watch)
  --lines           one line per reading instead of a full screen (watch); the default
                    when the output is a file or a pipe
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
    Watch { every: Duration, process: Option<String>, lines: bool },
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
    let (mut json, mut cpu, mut process, mut every, mut lines) = (false, false, None, Duration::from_secs(1), false);
    let mut i = 0;
    while i < rest.len() {
        let value = |i: usize| rest.get(i + 1).copied().ok_or_else(|| format!("{} needs a value", rest[i]));
        match rest[i] {
            "--json" => json = true,
            "--cpu" => cpu = true,
            "--lines" => lines = true,
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
        "watch" => &["--process", "--every", "--lines"],
        "target" => &["--json", "--cpu"],
        other => return Err(format!("unknown command `machine {other}`")),
    };
    if let Some(bad) = rest.iter().find(|w| w.starts_with("--") && !allowed.contains(w)) {
        return Err(format!("`machine {command}` does not take {bad}"));
    }
    Ok(match command {
        "info" => Cmd::Info { json },
        "usage" => Cmd::Usage { json, process },
        "watch" => Cmd::Watch { every, process, lines },
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
        Cmd::Watch { every, process, lines } => watch(every, process.as_deref(), lines)?,
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

/// Purpose: This function shows memory in use until the user quits or its output is
/// closed.
///
/// Inputs:
/// - `every`: the time between refreshes.
/// - `process`: a program name whose running copies to show.
/// - `lines`: `true` to write one line per reading, even to a terminal.
///
/// Outputs: `Ok(())` once the user presses q or the output is closed, as when piped into
/// `head`, or the message to print when reading the machine fails.
fn watch(
    every: Duration,
    process: Option<&str>,
    lines: bool,
) -> Result<(), String>
{
    let m = machine::current().map_err(|e| e.to_string())?;
    let start = Instant::now();

    // One line per reading, for a log file, a pipe, or anyone who asks with --lines.
    if lines || !std::io::stdout().is_terminal() {
        loop {
            let u = machine::usage(process).map_err(|e| e.to_string())?;
            let line = format!("{:>8.1}s  {}\n", start.elapsed().as_secs_f64(), usage_line(&u));
            if !write_out(&line) {
                return Ok(());
            }
            std::thread::sleep(every);
        }
    }

    // A full screen of its own, the way vim and top work. Dropping `screen` gives the
    // terminal back, on a quit and on an error alike.
    let chip = m.chip.as_deref().unwrap_or(&m.os);
    let mut screen = Screen::open();
    loop {
        let u = machine::usage(process).map_err(|e| e.to_string())?;
        let text = format!(
            "dovata machine watch: {chip}, every {} s, running {} s\n\n{}\nPress q to quit.",
            every.as_secs_f64(),
            start.elapsed().as_secs(),
            usage_text(&u, process)
        );
        if !write_out(&frame(&text)) || screen.wait(every) {
            return Ok(());
        }
    }
}

/// Purpose: This struct holds the terminal while `watch` shows its full screen, and gives
/// the terminal back as it was when it is dropped.
struct Screen {
    /// The terminal's settings from before, as `stty -g` prints them, or `None` when the
    /// keyboard is not a terminal.
    saved: Option<String>,
    /// Each key the user presses, read on a second thread, or `None` when there is no
    /// keyboard to read.
    keys: Option<mpsc::Receiver<u8>>,
}

impl Screen {
    /// Purpose: This function switches the terminal to a full screen of its own.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the [`Screen`].
    fn open() -> Screen
    {
        // Each key reaches dovata as soon as it is pressed and is not shown on screen.
        // Ctrl-C arrives as a key instead of stopping the program, so the terminal is
        // always given back. When stdin is not a terminal, its settings are left alone.
        let saved = if std::io::stdin().is_terminal() { stty(&["-g"]) } else { None };
        let saved = saved.filter(|_| stty(&["-icanon", "-echo", "-isig", "min", "1", "time", "0"]).is_some());
        let keys = saved.as_ref().map(|_| read_keys());
        // Switch to the terminal's second screen, and hide the cursor.
        write_out("\x1b[?1049h\x1b[?25l");
        Screen { saved, keys }
    }

    /// Purpose: This function waits until the next refresh, or until the user presses a
    /// key.
    ///
    /// Inputs:
    /// - `every`: the longest time to wait.
    ///
    /// Outputs: `true` when the user pressed q or Ctrl-C.
    fn wait(
        &mut self,
        every: Duration,
    ) -> bool
    {
        let Some(keys) = &self.keys else {
            std::thread::sleep(every);
            return false;
        };
        match keys.recv_timeout(every) {
            // Any other key refreshes the screen right away.
            Ok(key) => is_quit(key),
            Err(mpsc::RecvTimeoutError::Timeout) => false,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // The keyboard closed, so only the timer is left.
                self.keys = None;
                false
            }
        }
    }
}

impl Drop for Screen {
    /// Purpose: This function gives the terminal back as it was before `watch` started.
    ///
    /// Inputs: none.
    ///
    /// Outputs: none.
    fn drop(&mut self)
    {
        // Show the cursor, and switch back to the first screen, which still holds what
        // was there before.
        write_out("\x1b[?25h\x1b[?1049l");
        if let Some(settings) = &self.saved {
            stty(&[settings.as_str()]);
        }
    }
}

/// Purpose: This function runs `stty` on the terminal dovata was started from.
///
/// Inputs:
/// - `args`: the arguments for `stty`, such as `["-g"]`.
///
/// Outputs: what `stty` printed, trimmed, or `None` when it could not run or failed.
fn stty(
    args: &[&str],
) -> Option<String>
{
    // stty reads and changes the terminal on its own stdin, so it gets dovata's.
    let out = Command::new("stty").args(args).stdin(Stdio::inherit()).stderr(Stdio::null()).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// Purpose: This function starts a thread that reads each key the user presses.
///
/// Inputs: none.
///
/// Outputs: the receiving end, which gets one byte for each key.
fn read_keys() -> mpsc::Receiver<u8>
{
    let (send, keys) = mpsc::channel();
    std::thread::spawn(move || {
        let mut byte = [0u8; 1];
        // The thread ends when stdin closes, or with the program.
        while let Ok(1) = std::io::stdin().read(&mut byte) {
            if send.send(byte[0]).is_err() {
                break;
            }
        }
    });
    keys
}

/// Purpose: This function tells whether a key asks `watch` to quit.
///
/// Inputs:
/// - `key`: the byte the terminal sent for the key.
///
/// Outputs: `true` for q, Q and Ctrl-C.
fn is_quit(
    key: u8,
) -> bool
{
    matches!(key, b'q' | b'Q' | 0x03)
}

/// Purpose: This function turns text into one full-screen frame that draws over the last
/// one in place.
///
/// Inputs:
/// - `text`: the lines to show.
///
/// Outputs: the text, with the terminal codes that draw it from the top left.
fn frame(
    text: &str,
) -> String
{
    // Start at the top left, clear the rest of each line after its text, and clear
    // everything below the last line. Nothing is cleared before it is drawn over, so the
    // screen does not flicker, and the last line has no newline, so it never scrolls.
    let lines: Vec<String> = text.lines().map(|l| format!("{l}\x1b[K")).collect();
    format!("\x1b[H{}\x1b[J", lines.join("\n"))
}

/// Purpose: This function writes text to the terminal or pipe right away.
///
/// Inputs:
/// - `text`: the text.
///
/// Outputs: `false` when the output is closed, as when piped into `head`.
fn write_out(
    text: &str,
) -> bool
{
    let mut stdout = std::io::stdout().lock();
    // Rust ignores SIGPIPE, so a closed pipe shows up here as a write error.
    stdout.write_all(text.as_bytes()).and_then(|()| stdout.flush()).is_ok()
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
    t: &dovata::target::Target,
) -> String
{
    use dovata::target::Target;
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
        assert_eq!(p("watch --every 0.5"), Ok(Cmd::Watch { every: Duration::from_millis(500), process: None, lines: false }));
        assert_eq!(p("watch --lines"), Ok(Cmd::Watch { every: Duration::from_secs(1), process: None, lines: true }));
        assert_eq!(p("target --cpu"), Ok(Cmd::Target { json: false, cpu: true }));
        assert_eq!(p("target --json"), Ok(Cmd::Target { json: true, cpu: false }));
    }

    #[test]
    fn bad_command_lines_are_refused()
    {
        for bad in ["nope", "info --cpu", "usage --process", "watch --every soon", "watch --every 0", "info --verbose", "usage --lines"] {
            assert!(p(bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn a_frame_draws_over_the_last_one_in_place()
    {
        assert_eq!(frame("a\nb\n"), "\x1b[Ha\x1b[K\nb\x1b[K\x1b[J");
    }

    #[test]
    fn q_and_ctrl_c_quit_and_other_keys_do_not()
    {
        assert!(is_quit(b'q') && is_quit(b'Q') && is_quit(0x03));
        assert!(!is_quit(b' ') && !is_quit(b'j') && !is_quit(0x1b));
    }

    #[test]
    fn target_text_shows_units()
    {
        let apple = dovata::target::Target::Apple { unified_bytes: 19_069_665_280, max_buffer_bytes: 14_302_248_960, page_bytes: 16384 };
        assert_eq!(
            target_block(&apple),
            "apple target\n  memory          17.8 GiB the plan may use\n  largest buffer  13.3 GiB\n  page size       16 KiB\n"
        );
        let cuda = dovata::target::Target::Cuda { vram_bytes: 80 << 30 };
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
