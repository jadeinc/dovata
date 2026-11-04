//! Author: Joe Ricks
//! File: src/cli/host.rs
//! Description: The `dovata start`, `status` and `kill` commands, which run a model in a
//! process of its own, report on it and stop it, and `dovata host`, the process itself.

use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use dovata::host::{self, HostError, Holds, Hosted, StartLine, StartOptions, State, Status};
use dovata::target::RegionKind;

use super::{json, out, plan, size, target_text, Fail};

/// The help text for `dovata start`.
pub(crate) const START_HELP: &str = "\
usage: dovata start <model> [options]

Load a model into memory in a process of its own, which keeps running until
`dovata kill` stops it. Programs map its memory through the dovata library.

<model> is a GGUF file, or a Hugging Face link to a repo or to one file in it.

options:
  --name NAME          the process's name; the default comes from the model, such as
                       qwen3-8b
  --cpu                hold the model for the CPU, even when this machine has a GPU
  --foreground         run in this terminal instead of in the background; Ctrl-C stops it
  --json               print the status as JSON once the model is loaded

The model and step options of `dovata plan` apply too: --quant, --allow-requantize,
--layout, --ctx, --tokens, --outputs, --kv, --kv-k and --kv-v. See `dovata plan --help`.
";

/// The help text for `dovata status`.
pub(crate) const STATUS_HELP: &str = "\
usage: dovata status [NAME] [--json]

Show a running model's memory, region by region, and how much of it is resident in
physical memory right now. With no name, show every running model.
";

/// The help text for `dovata kill`.
pub(crate) const KILL_HELP: &str = "\
usage: dovata kill [NAME | --all]

Stop a running model. With no name, stop the only one running.
";

/// Purpose: This struct holds the options of one `dovata start` command line.
#[derive(Debug, Clone, PartialEq)]
struct StartOpts {
    /// The model and step options, read the way `dovata plan` reads them.
    plan: plan::Opts,
    /// `--name`.
    name: Option<String>,
    /// `--cpu`.
    cpu: bool,
    /// `--foreground`.
    foreground: bool,
}

/// Purpose: This function runs `dovata start`.
///
/// Inputs:
/// - `args`: the arguments after `start`.
///
/// Outputs: `Ok(())` once the model is loaded, or after a foreground process stops; or a
/// [`Fail`].
pub(crate) fn start(
    args: &[String],
) -> Result<(), Fail>
{
    if args.is_empty() || args.iter().any(|a| matches!(a.as_str(), "help" | "--help" | "-h")) {
        out(START_HELP);
        return Ok(());
    }
    let o = parse_start(args).map_err(|msg| Fail::Usage(format!("{msg}\n\nRun `dovata start --help` for the options.")))?;
    let parts = plan::model_files(&o.plan)?;
    let program = std::env::current_exe().map_err(|e| Fail::Run(format!("cannot find this program's path: {e}")))?;
    let opts = StartOptions { name: o.name.clone(), layout: o.plan.layout.clone(), step: o.plan.step, cpu: o.cpu, program: Some(program), ..StartOptions::default() };
    let spec = host::spec(&parts, &opts).map_err(fail)?;
    if o.foreground {
        return host::serve(&spec, |r| {
            out(&format!("{} is running, pid {}. Stop it with Ctrl-C, or with `dovata kill {}`.\n", r.name, r.pid, r.name));
            true
        })
        .map_err(fail);
    }
    if host::attach(&spec.name).is_ok() {
        return Err(fail(HostError::AlreadyRunning(spec.name)));
    }
    eprintln!("dovata: loading {} into memory as {}", parts[0].display(), spec.name);
    let opts = StartOptions { name: Some(spec.name), ..opts };
    let h = host::start_parts(&parts, &opts).map_err(fail)?;
    let s = h.status().map_err(fail)?;
    if o.plan.json {
        out(&format!("{}\n", json(&s)));
    } else {
        out(&status_text(&s, now()));
        let log = host::run_dir().map(|d| host::log_path(&d, &s.name).display().to_string()).unwrap_or_default();
        out(&format!("Stop it with `dovata kill {}`. Its log is {log}.\n", s.name));
    }
    Ok(())
}

/// Purpose: This function reads the arguments of `dovata start`.
///
/// Inputs:
/// - `args`: the arguments after `start`.
///
/// Outputs: the [`StartOpts`], or a message saying what is wrong with them.
fn parse_start(
    args: &[String],
) -> Result<StartOpts, String>
{
    let (mut name, mut cpu, mut foreground) = (None, false, false);
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--name" => {
                let n = args.get(i + 1).ok_or("--name needs a value")?;
                host::check_name(n).map_err(|e| e.to_string())?;
                name = Some(n.clone());
                i += 1;
            }
            "--cpu" => cpu = true,
            "--foreground" => foreground = true,
            "--target" | "--memory" | "--max-buffer" | "--page" => {
                return Err(format!(
                    "dovata start holds the model on this machine, so it takes no {}; add --cpu to hold it for the CPU instead of the GPU",
                    args[i]
                ))
            }
            _ => rest.push(args[i].clone()),
        }
        i += 1;
    }
    let plan = plan::parse(&rest)?;
    if foreground && plan.json {
        return Err("--json prints the status once a background start is ready, so it does not go with --foreground".into());
    }
    Ok(StartOpts { plan, name, cpu, foreground })
}

/// Purpose: This function runs `dovata status`.
///
/// Inputs:
/// - `args`: the arguments after `status`.
///
/// Outputs: `Ok(())`, or a [`Fail`].
pub(crate) fn status(
    args: &[String],
) -> Result<(), Fail>
{
    let (mut as_json, mut name) = (false, None);
    for a in args {
        match a.as_str() {
            "help" | "--help" | "-h" => {
                out(STATUS_HELP);
                return Ok(());
            }
            "--json" => as_json = true,
            f if f.starts_with("--") => return Err(Fail::Usage(format!("unknown option {f:?}\n\n{STATUS_HELP}"))),
            n if name.is_none() => name = Some(n.to_owned()),
            n => return Err(Fail::Usage(format!("one name at a time; {n:?} is a second one"))),
        }
    }
    let hosted = match &name {
        Some(n) => vec![host::attach(n).map_err(fail)?],
        None => host::running().map_err(fail)?,
    };
    let statuses = hosted.iter().map(Hosted::status).collect::<Result<Vec<_>, _>>().map_err(fail)?;
    if as_json {
        out(&format!("{}\n", json(&statuses)));
    } else if statuses.is_empty() {
        out("No dovata process is running. Start one with `dovata start <model>`.\n");
    } else {
        let now = now();
        let texts: Vec<String> = statuses.iter().map(|s| status_text(s, now)).collect();
        out(&texts.join("\n"));
    }
    Ok(())
}

/// Purpose: This function runs `dovata kill`.
///
/// Inputs:
/// - `args`: the arguments after `kill`.
///
/// Outputs: `Ok(())` once every named process has stopped, or a [`Fail`].
pub(crate) fn kill(
    args: &[String],
) -> Result<(), Fail>
{
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    let targets = match words.as_slice() {
        ["help" | "--help" | "-h", ..] => {
            out(KILL_HELP);
            return Ok(());
        }
        ["--all"] => host::running().map_err(fail)?,
        [name] if !name.starts_with("--") => vec![host::attach(name).map_err(fail)?],
        [] => {
            let all = host::running().map_err(fail)?;
            match all.len() {
                0 => return Err(Fail::Run("no dovata process is running".into())),
                1 => all,
                _ => {
                    let names: Vec<&str> = all.iter().map(Hosted::name).collect();
                    return Err(Fail::Usage(format!("{} are running; name one, or use --all", names.join(", "))));
                }
            }
        }
        _ => return Err(Fail::Usage(format!("give one name, or --all\n\n{KILL_HELP}"))),
    };
    for h in &targets {
        h.kill().map_err(fail)?;
        out(&format!("{} stopped\n", h.name()));
    }
    Ok(())
}

/// Purpose: This function runs `dovata host`, the process `dovata start` and the
/// library's `host::start` run in the background. It takes the model's spec as JSON in
/// its one argument and writes one line to standard output: that it is ready, or why
/// not.
///
/// Inputs:
/// - `args`: the arguments after `host`: the spec.
///
/// Outputs: `Ok(())` after a stop request, or a [`Fail`] when the model cannot be held.
pub(crate) fn serve(
    args: &[String],
) -> Result<(), Fail>
{
    let result = match args {
        [text] => serde_json::from_str::<host::Spec>(text).map_err(|e| HostError::Protocol(format!("the spec: {e}"))),
        _ => Err(HostError::Protocol("`dovata host` takes one argument, the spec as JSON; people use `dovata start`".into())),
    }
    .and_then(|spec| host::serve(&spec, |r| say(&StartLine::Ready(r.clone()))));
    match result {
        Ok(()) => Ok(()),
        Err(e) => {
            say(&StartLine::Error(e.to_string()));
            Err(fail(e))
        }
    }
}

/// Purpose: This function writes the start line for the program that started this one.
///
/// Inputs:
/// - `line`: the line.
///
/// Outputs: `true` when it was written, `false` when nobody is reading any more.
fn say(
    line: &StartLine,
) -> bool
{
    let text = serde_json::to_string(line).unwrap_or_default();
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{text}").and_then(|()| stdout.flush()).is_ok()
}

/// Purpose: This function turns a [`HostError`] into the command's failure.
///
/// Inputs:
/// - `e`: the error.
///
/// Outputs: [`Fail::Usage`] for a bad name, else [`Fail::Run`].
fn fail(
    e: HostError,
) -> Fail
{
    match e {
        HostError::BadName(_) => Fail::Usage(e.to_string()),
        _ => Fail::Run(e.to_string()),
    }
}

/// Purpose: This function reads every running model's status, for the `dovata machine`
/// readings.
///
/// Inputs: none.
///
/// Outputs: each one that answered, sorted by name; none when the run directory cannot be
/// read.
pub(crate) fn statuses() -> Vec<Status>
{
    host::running().unwrap_or_default().iter().filter_map(|h| h.status().ok()).collect()
}

/// Purpose: This function gives the time now.
///
/// Inputs: none.
///
/// Outputs: seconds since 1970.
fn now() -> u64
{
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Purpose: This function formats a running model's status for a person.
///
/// Inputs:
/// - `s`: the status.
/// - `now`: the time now, in seconds since 1970.
///
/// Outputs: the text, one item per line.
pub(crate) fn status_text(
    s: &Status,
    now: u64,
) -> String
{
    let ftype = s.model.file_type.as_deref().map(|f| format!(", {f}")).unwrap_or_default();
    let mut t = format!("{}, pid {}, up {}, dovata {}\n", s.name, s.pid, duration(now.saturating_sub(s.started_unix)), s.version);
    t += &format!("  model      {}, {}{ftype}\n", s.model.name.as_deref().unwrap_or("unnamed model"), s.model.architecture);
    t += &format!("  machine    {}\n", target_text(&s.target));
    t += &match s.state {
        State::Ready => "  state      ready\n".to_owned(),
        State::Loading { loaded_bytes, total_bytes } => format!("  state      loading, {} of {}\n", size(loaded_bytes), size(total_bytes)),
    };
    for f in &s.files {
        t += &format!("  file       {}\n             {} resident of {}\n", f.path.display(), size(f.resident_bytes), size(f.bytes));
    }
    t += "  regions\n";
    for r in &s.regions {
        t += &format!("    {:<3} {:<8} {:>10}  {:>10} resident  {}\n", r.id, kind(r.kind), size(r.bytes), size(r.resident_bytes), holds_text(&r.holds));
    }
    t += &format!("  memory     {} resident of {}, the file once and every allocation\n", size(s.resident_bytes()), size(s.bytes()));
    if s.clients.is_empty() {
        t += "  mapped by  no program yet\n";
    }
    for c in &s.clients {
        t += &format!("  mapped by  pid {}, {}\n", c.pid, if c.alive { "running" } else { "exited" });
    }
    t += &format!("  table hash {}\n", s.table_hash);
    t
}

/// Purpose: This function formats one running model as one line, for `dovata machine
/// usage` and `watch`.
///
/// Inputs:
/// - `s`: the status.
///
/// Outputs: the line, without a newline.
pub(crate) fn status_line(
    s: &Status,
) -> String
{
    match s.state {
        State::Ready => format!("{}, pid {}, {} resident of {}", s.name, s.pid, size(s.resident_bytes()), size(s.bytes())),
        State::Loading { loaded_bytes, total_bytes } => format!("{}, pid {}, loading, {} of {}", s.name, s.pid, size(loaded_bytes), size(total_bytes)),
    }
}

/// Purpose: This function names the kind of memory a region is.
///
/// Inputs:
/// - `k`: the kind.
///
/// Outputs: a name such as `mmap` or `unified`.
pub(crate) fn kind(
    k: RegionKind,
) -> &'static str
{
    match k {
        RegionKind::Mmap => "mmap",
        RegionKind::Host => "host",
        RegionKind::Vram => "vram",
        RegionKind::Unified => "unified",
    }
}

/// Purpose: This function says what a region holds.
///
/// Inputs:
/// - `holds`: what it holds.
///
/// Outputs: a phrase such as `weights` or `KV cache`.
fn holds_text(
    holds: &[Holds],
) -> String
{
    let words: Vec<&str> = holds
        .iter()
        .map(|h| match h {
            Holds::Weights => "weights",
            Holds::KvCache => "KV cache",
            Holds::State => "recurrent state",
            Holds::Step => "step inputs, scratch and outputs",
        })
        .collect();
    words.join(", ")
}

/// Purpose: This function formats a span of time the way a person says it.
///
/// Inputs:
/// - `secs`: the span in seconds.
///
/// Outputs: text such as `45 s`, `12 min`, `3 h 5 min` or `2 d 4 h`.
fn duration(
    secs: u64,
) -> String
{
    match secs {
        s if s < 60 => format!("{s} s"),
        s if s < 3600 => format!("{} min", s / 60),
        s if s < 86_400 => format!("{} h {} min", s / 3600, s % 3600 / 60),
        s => format!("{} d {} h", s / 86_400, s % 86_400 / 3600),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use dovata::host::{Client, FileUse, ModelInfo, RegionUse};
    use dovata::target::Target;

    /// Purpose: This function reads a `dovata start` command line written as one string.
    ///
    /// Inputs:
    /// - `line`: the arguments after `start`, separated by spaces.
    ///
    /// Outputs: what [`parse_start`] returns for them.
    fn p(
        line: &str,
    ) -> Result<StartOpts, String>
    {
        parse_start(&line.split_whitespace().map(str::to_owned).collect::<Vec<_>>())
    }

    #[test]
    fn start_takes_plan_options_and_its_own()
    {
        let o = p("m.gguf --name qwen --cpu --ctx 8192 --kv q8_0").unwrap();
        assert_eq!(o.plan.model, "m.gguf");
        assert_eq!((o.name.as_deref(), o.cpu, o.foreground), (Some("qwen"), true, false));
        assert_eq!(o.plan.step.n_ctx, 8192);
        assert!(p("m.gguf --foreground").unwrap().foreground);
        for bad in ["", "m.gguf --name", "m.gguf --name Bad/Name", "m.gguf --target cpu", "m.gguf --memory 8G", "m.gguf --foreground --json", "m.gguf --nope"] {
            assert!(p(bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn status_shows_every_region_and_who_mapped_it()
    {
        let s = Status {
            name: "qwen3-8b".into(),
            pid: 4121,
            version: "3.1.0".into(),
            state: State::Ready,
            started_unix: 1000,
            model: ModelInfo { name: Some("Qwen3-8B".into()), architecture: "qwen3".into(), file_type: Some("Q4_K_M".into()) },
            target: Target::Apple { unified_bytes: 19_069_665_280, max_buffer_bytes: 14_302_248_960, page_bytes: 16384 },
            table_hash: "e9f5".into(),
            files: vec![FileUse { path: PathBuf::from("/m/Qwen3-8B-Q4_K_M.gguf"), bytes: 5 << 30, resident_bytes: 4 << 30 }],
            regions: vec![
                RegionUse { id: 0, kind: RegionKind::Mmap, holds: vec![Holds::Weights], bytes: 5 << 30, resident_bytes: 4 << 30 },
                RegionUse { id: 1, kind: RegionKind::Unified, holds: vec![Holds::KvCache], bytes: 576 << 20, resident_bytes: 576 << 20 },
                RegionUse { id: 2, kind: RegionKind::Unified, holds: vec![Holds::Step], bytes: 300 << 20, resident_bytes: 300 << 20 },
            ],
            clients: vec![Client { pid: 5120, mapped_unix: 1100, alive: true }, Client { pid: 5121, mapped_unix: 1200, alive: false }],
        };
        let t = status_text(&s, 1000 + 3 * 3600 + 5 * 60);
        assert!(t.starts_with("qwen3-8b, pid 4121, up 3 h 5 min, dovata 3.1.0\n"), "{t}");
        assert!(t.contains("model      Qwen3-8B, qwen3, Q4_K_M"), "{t}");
        assert!(t.contains("state      ready"), "{t}");
        assert!(t.contains("4.0 GiB resident of 5.0 GiB"), "{t}");
        assert!(t.contains("1   unified   576.0 MiB   576.0 MiB resident  KV cache"), "{t}");
        assert!(t.contains("memory     4.9 GiB resident of 5.9 GiB"), "{t}");
        assert!(t.contains("mapped by  pid 5120, running") && t.contains("pid 5121, exited"), "{t}");
        assert_eq!(status_line(&s), "qwen3-8b, pid 4121, 4.9 GiB resident of 5.9 GiB");
        let loading = Status { state: State::Loading { loaded_bytes: 1 << 30, total_bytes: 6 << 30 }, clients: vec![], ..s };
        assert!(status_text(&loading, 1000).contains("state      loading, 1.0 GiB of 6.0 GiB"));
        assert!(status_text(&loading, 1000).contains("mapped by  no program yet"));
        assert_eq!(status_line(&loading), "qwen3-8b, pid 4121, loading, 1.0 GiB of 6.0 GiB");
    }

    #[test]
    fn spans_read_the_way_a_person_says_them()
    {
        assert_eq!(duration(45), "45 s");
        assert_eq!(duration(12 * 60 + 5), "12 min");
        assert_eq!(duration(3 * 3600 + 5 * 60), "3 h 5 min");
        assert_eq!(duration(2 * 86_400 + 4 * 3600), "2 d 4 h");
    }
}
