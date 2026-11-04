//! Author: Joe Ricks
//! File: src/cli/host.rs
//! Description: The `dovata start`, `add`, `remove`, `status` and `kill` commands, which
//! run the dovata process, put models in it and take them out, report on it and stop it,
//! and `dovata host`, the process itself.

use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use dovata::host::{self, AddOptions, HostError, Holds, ModelStatus, StartLine, StartOptions, State, Status};
use dovata::target::RegionKind;

use super::{json, out, plan, size, target_text, Fail};

/// The help text for `dovata start`.
pub(crate) const START_HELP: &str = "\
usage: dovata start [--foreground]

Start the dovata process, which holds models in memory until `dovata kill` stops it.
It starts with no models; `dovata add` puts one in. There is one dovata process per
user, so a second start finds the first one running.

options:
  --foreground   run in this terminal instead of in the background; Ctrl-C stops it
";

/// The help text for `dovata add`.
pub(crate) const ADD_HELP: &str = "\
usage: dovata add <model> [options]

Load a model into the running dovata process, and wait until it is in memory. Programs
map its memory through the dovata library.

<model> is a GGUF file, or a Hugging Face link to a repo or to one file in it.

options:
  --name NAME   the model's name in dovata; the default comes from the model, such as
                qwen3-8b
  --cpu         hold the model for the CPU, even when this machine has a GPU
  --json        print the model's status as JSON once it is loaded

The model and step options of `dovata plan` apply too: --quant, --allow-requantize,
--layout, --ctx, --tokens, --outputs, --kv, --kv-k and --kv-v. See `dovata plan --help`.
";

/// The help text for `dovata remove`.
pub(crate) const REMOVE_HELP: &str = "\
usage: dovata remove <name>

Take a model out of the dovata process. Programs that mapped its memory keep it until
they unmap it.
";

/// The help text for `dovata status`.
pub(crate) const STATUS_HELP: &str = "\
usage: dovata status [NAME] [--json]

Show the dovata process and each model in it: its memory region by region, how much of
it is resident in physical memory right now, and the programs that mapped it. With a
name, show that model alone.
";

/// The help text for `dovata kill`.
pub(crate) const KILL_HELP: &str = "\
usage: dovata kill

Stop the dovata process, with every model in it.
";

/// Purpose: This struct holds the options of one `dovata add` command line.
#[derive(Debug, Clone, PartialEq)]
struct AddOpts {
    /// The model and step options, read the way `dovata plan` reads them.
    plan: plan::Opts,
    /// `--name`.
    name: Option<String>,
    /// `--cpu`.
    cpu: bool,
}

/// Purpose: This function says whether a command line asks for help.
///
/// Inputs:
/// - `args`: the arguments after the command.
///
/// Outputs: `true` when one of them is `help`, `--help` or `-h`.
fn wants_help(
    args: &[String],
) -> bool
{
    args.iter().any(|a| matches!(a.as_str(), "help" | "--help" | "-h"))
}

/// Purpose: This function runs `dovata start`.
///
/// Inputs:
/// - `args`: the arguments after `start`.
///
/// Outputs: `Ok(())` once the process runs, or after a foreground process stops; or a
/// [`Fail`].
pub(crate) fn start(
    args: &[String],
) -> Result<(), Fail>
{
    if wants_help(args) {
        out(START_HELP);
        return Ok(());
    }
    let foreground = match args {
        [] => false,
        [f] if f == "--foreground" => true,
        _ => return Err(Fail::Usage(format!("`dovata start` takes no model and no options but --foreground\n\n{START_HELP}"))),
    };
    if let Ok(s) = host::connect().and_then(|p| p.status()) {
        out(&format!("dovata is already running, pid {}, with {}.\n", s.pid, count(s.models.len())));
        return Ok(());
    }
    if foreground {
        return host::serve(|r| {
            out(&format!(
                "dovata is running, pid {}. Add a model with `dovata add <model>` from another terminal; Ctrl-C or `dovata kill` stops it.\n",
                r.pid
            ));
            true
        })
        .map_err(fail);
    }
    let program = std::env::current_exe().map_err(|e| Fail::Run(format!("cannot find this program's path: {e}")))?;
    let p = host::start(&StartOptions { program: Some(program) }).map_err(fail)?;
    let s = p.status().map_err(fail)?;
    let log = host::run_dir().map(|d| host::log_path(&d).display().to_string()).unwrap_or_default();
    out(&format!("dovata is running, pid {}. Add a model with `dovata add <model>`.\nIts log is {log}.\n", s.pid));
    Ok(())
}

/// Purpose: This function runs `dovata add`.
///
/// Inputs:
/// - `args`: the arguments after `add`.
///
/// Outputs: `Ok(())` once the model is in memory, or a [`Fail`].
pub(crate) fn add(
    args: &[String],
) -> Result<(), Fail>
{
    if args.is_empty() || wants_help(args) {
        out(ADD_HELP);
        return Ok(());
    }
    let o = parse_add(args).map_err(|msg| Fail::Usage(format!("{msg}\n\nRun `dovata add --help` for the options.")))?;
    // Found first, so a stopped process is reported before anything is downloaded.
    let p = host::connect().map_err(fail)?;
    let parts = plan::model_files(&o.plan)?;
    let opts = AddOptions { name: o.name.clone(), layout: o.plan.layout.clone(), step: o.plan.step, cpu: o.cpu, ..AddOptions::default() };
    let spec = host::spec(&parts, &opts).map_err(fail)?;
    eprintln!("dovata: adding {} as {}", parts[0].display(), spec.name);
    let m = p.add_parts(&parts, &AddOptions { name: Some(spec.name), ..opts }).map_err(fail)?;
    if o.plan.json {
        out(&format!("{}\n", json(&m)));
    } else {
        out(&model_text(&m, now()));
    }
    Ok(())
}

/// Purpose: This function reads the arguments of `dovata add`.
///
/// Inputs:
/// - `args`: the arguments after `add`.
///
/// Outputs: the [`AddOpts`], or a message saying what is wrong with them.
fn parse_add(
    args: &[String],
) -> Result<AddOpts, String>
{
    let (mut name, mut cpu) = (None, false);
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
            "--target" | "--memory" | "--max-buffer" | "--page" => {
                return Err(format!(
                    "dovata holds models on this machine, so `dovata add` takes no {}; add --cpu to hold the model for the CPU instead of the GPU",
                    args[i]
                ))
            }
            _ => rest.push(args[i].clone()),
        }
        i += 1;
    }
    Ok(AddOpts { plan: plan::parse(&rest)?, name, cpu })
}

/// Purpose: This function runs `dovata remove`.
///
/// Inputs:
/// - `args`: the arguments after `remove`.
///
/// Outputs: `Ok(())` once the model is out, or a [`Fail`].
pub(crate) fn remove(
    args: &[String],
) -> Result<(), Fail>
{
    let name = match args {
        [] => {
            out(REMOVE_HELP);
            return Ok(());
        }
        _ if wants_help(args) => {
            out(REMOVE_HELP);
            return Ok(());
        }
        [name] if !name.starts_with("--") => name,
        _ => return Err(Fail::Usage(format!("give one model's name\n\n{REMOVE_HELP}"))),
    };
    host::connect().and_then(|p| p.remove(name)).map_err(fail)?;
    out(&format!("{name} removed\n"));
    Ok(())
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
    if wants_help(args) {
        out(STATUS_HELP);
        return Ok(());
    }
    let (mut as_json, mut name) = (false, None);
    for a in args {
        match a.as_str() {
            "--json" => as_json = true,
            f if f.starts_with("--") => return Err(Fail::Usage(format!("unknown option {f:?}\n\n{STATUS_HELP}"))),
            n if name.is_none() => name = Some(n.to_owned()),
            n => return Err(Fail::Usage(format!("one name at a time; {n:?} is a second one"))),
        }
    }
    let s = match host::connect().and_then(|p| p.status()) {
        Ok(s) => s,
        Err(HostError::NotRunning) if name.is_none() && !as_json => {
            out("dovata is not running. Start it with `dovata start`.\n");
            return Ok(());
        }
        Err(e) => return Err(fail(e)),
    };
    match name {
        Some(n) => {
            let m = s.model(&n).ok_or_else(|| Fail::Run(format!("dovata holds no model named {n}")))?;
            out(&if as_json { format!("{}\n", json(m)) } else { model_text(m, now()) });
        }
        None if as_json => out(&format!("{}\n", json(&s))),
        None => out(&status_text(&s, now())),
    }
    Ok(())
}

/// Purpose: This function runs `dovata kill`.
///
/// Inputs:
/// - `args`: the arguments after `kill`.
///
/// Outputs: `Ok(())` once the process has stopped, or a [`Fail`].
pub(crate) fn kill(
    args: &[String],
) -> Result<(), Fail>
{
    if wants_help(args) {
        out(KILL_HELP);
        return Ok(());
    }
    if !args.is_empty() {
        return Err(Fail::Usage(format!(
            "`dovata kill` stops the whole process and takes no arguments; `dovata remove <name>` takes out one model\n\n{KILL_HELP}"
        )));
    }
    host::connect().and_then(|p| p.kill()).map_err(fail)?;
    out("dovata stopped\n");
    Ok(())
}

/// Purpose: This function runs `dovata host`, the process `dovata start` and the
/// library's `host::start` run in the background. It writes one line to standard
/// output: that it is ready, or why not.
///
/// Inputs:
/// - `args`: the arguments after `host`; there should be none.
///
/// Outputs: `Ok(())` after a stop request, or a [`Fail`] when the process cannot start.
pub(crate) fn serve(
    args: &[String],
) -> Result<(), Fail>
{
    let result = if args.is_empty() {
        host::serve(|r| say(&StartLine::Ready(r.clone())))
    } else {
        Err(HostError::Protocol("`dovata host` takes no arguments; people use `dovata start`".into()))
    };
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

/// Purpose: This function reads the dovata process's status, for the `dovata machine`
/// readings.
///
/// Inputs: none.
///
/// Outputs: the status, or `None` when dovata is not running.
pub(crate) fn current() -> Option<Status>
{
    host::connect().and_then(|p| p.status()).ok()
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

/// Purpose: This function says how many models there are, in words.
///
/// Inputs:
/// - `n`: the count.
///
/// Outputs: text such as `no models`, `1 model` or `3 models`.
fn count(
    n: usize,
) -> String
{
    match n {
        0 => "no models".into(),
        1 => "1 model".into(),
        n => format!("{n} models"),
    }
}

/// Purpose: This function formats the dovata process's status for a person: one line for
/// the process, then each model.
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
    let mut t = format!("{}\n", process_line(s, now));
    if s.models.is_empty() {
        t += "No models. Add one with `dovata add <model>`.\n";
    }
    for m in &s.models {
        t += "\n";
        t += &model_text(m, now);
    }
    t
}

/// Purpose: This function describes the dovata process in one line.
///
/// Inputs:
/// - `s`: the status.
/// - `now`: the time now, in seconds since 1970.
///
/// Outputs: the line, without a newline.
fn process_line(
    s: &Status,
    now: u64,
) -> String
{
    format!(
        "dovata {}, pid {}, up {}, {}, {} resident of {}",
        s.version,
        s.pid,
        duration(now.saturating_sub(s.started_unix)),
        count(s.models.len()),
        size(s.resident_bytes()),
        size(s.bytes())
    )
}

/// Purpose: This function formats one model's status for a person.
///
/// Inputs:
/// - `m`: the model's status.
/// - `now`: the time now, in seconds since 1970.
///
/// Outputs: the text, one item per line.
pub(crate) fn model_text(
    m: &ModelStatus,
    now: u64,
) -> String
{
    let ftype = m.model.file_type.as_deref().map(|f| format!(", {f}")).unwrap_or_default();
    let mut t = format!("{}, added {} ago\n", m.name, duration(now.saturating_sub(m.added_unix)));
    t += &format!("  model      {}, {}{ftype}\n", m.model.name.as_deref().unwrap_or("unnamed model"), m.model.architecture);
    t += &format!("  machine    {}\n", target_text(&m.target));
    t += &match m.state {
        State::Ready => "  state      ready\n".to_owned(),
        State::Loading { loaded_bytes, total_bytes } => format!("  state      loading, {} of {}\n", size(loaded_bytes), size(total_bytes)),
    };
    for f in &m.files {
        t += &format!("  file       {}\n             {} resident of {}\n", f.path.display(), size(f.resident_bytes), size(f.bytes));
    }
    t += "  regions\n";
    for r in &m.regions {
        t += &format!("    {:<3} {:<8} {:>10}  {:>10} resident  {}\n", r.id, kind(r.kind), size(r.bytes), size(r.resident_bytes), holds_text(&r.holds));
    }
    t += &format!("  memory     {} resident of {}, the file once and every allocation\n", size(m.resident_bytes()), size(m.bytes()));
    if m.clients.is_empty() {
        t += "  mapped by  no program yet\n";
    }
    for c in &m.clients {
        t += &format!("  mapped by  pid {}, {}\n", c.pid, if c.alive { "running" } else { "exited" });
    }
    t += &format!("  table hash {}\n", m.table_hash);
    t
}

/// Purpose: This function formats the dovata process for `dovata machine usage` and
/// `watch`: one line for the process and one per model.
///
/// Inputs:
/// - `s`: the status.
///
/// Outputs: the lines, each ending in a newline.
pub(crate) fn usage_lines(
    s: &Status,
) -> String
{
    let mut t = format!("  dovata     pid {}, {}, {} resident of {}\n", s.pid, count(s.models.len()), size(s.resident_bytes()), size(s.bytes()));
    for m in &s.models {
        t += &match m.state {
            State::Ready => format!("  model      {}, {} resident of {}\n", m.name, size(m.resident_bytes()), size(m.bytes())),
            State::Loading { loaded_bytes, total_bytes } => format!("  model      {}, loading, {} of {}\n", m.name, size(loaded_bytes), size(total_bytes)),
        };
    }
    t
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
    use dovata::host::{Client, FileUse, ModelInfo, RegionUse};
    use dovata::target::Target;
    use std::path::PathBuf;

    /// Purpose: This function reads a `dovata add` command line written as one string.
    ///
    /// Inputs:
    /// - `line`: the arguments after `add`, separated by spaces.
    ///
    /// Outputs: what [`parse_add`] returns for them.
    fn p(
        line: &str,
    ) -> Result<AddOpts, String>
    {
        parse_add(&line.split_whitespace().map(str::to_owned).collect::<Vec<_>>())
    }

    /// Purpose: This function builds a model's status for the text tests.
    ///
    /// Inputs:
    /// - `name`: the model's name.
    ///
    /// Outputs: a ready model with three regions and two programs that mapped it.
    fn model(
        name: &str,
    ) -> ModelStatus
    {
        ModelStatus {
            name: name.into(),
            added_unix: 1000,
            state: State::Ready,
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
        }
    }

    #[test]
    fn add_takes_plan_options_and_its_own()
    {
        let o = p("m.gguf --name qwen --cpu --ctx 8192 --kv q8_0").unwrap();
        assert_eq!(o.plan.model, "m.gguf");
        assert_eq!((o.name.as_deref(), o.cpu), (Some("qwen"), true));
        assert_eq!(o.plan.step.n_ctx, 8192);
        for bad in ["", "m.gguf --name", "m.gguf --name Bad/Name", "m.gguf --target cpu", "m.gguf --memory 8G", "m.gguf --foreground", "m.gguf --nope"] {
            assert!(p(bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn start_and_kill_take_no_model()
    {
        let args = |line: &str| line.split_whitespace().map(str::to_owned).collect::<Vec<_>>();
        assert!(matches!(start(&args("m.gguf")), Err(Fail::Usage(_))));
        assert!(matches!(start(&args("--foreground --cpu")), Err(Fail::Usage(_))));
        assert!(matches!(kill(&args("qwen")), Err(Fail::Usage(m)) if m.contains("dovata remove")));
        assert!(matches!(remove(&args("a b")), Err(Fail::Usage(_))));
    }

    #[test]
    fn a_model_shows_every_region_and_who_mapped_it()
    {
        let m = model("qwen3-8b");
        let t = model_text(&m, 1000 + 3 * 3600 + 5 * 60);
        assert!(t.starts_with("qwen3-8b, added 3 h 5 min ago\n"), "{t}");
        assert!(t.contains("model      Qwen3-8B, qwen3, Q4_K_M"), "{t}");
        assert!(t.contains("state      ready"), "{t}");
        assert!(t.contains("4.0 GiB resident of 5.0 GiB"), "{t}");
        assert!(t.contains("1   unified   576.0 MiB   576.0 MiB resident  KV cache"), "{t}");
        assert!(t.contains("memory     4.9 GiB resident of 5.9 GiB"), "{t}");
        assert!(t.contains("mapped by  pid 5120, running") && t.contains("pid 5121, exited"), "{t}");
        let loading = ModelStatus { state: State::Loading { loaded_bytes: 1 << 30, total_bytes: 6 << 30 }, clients: vec![], ..m };
        assert!(model_text(&loading, 1000).contains("state      loading, 1.0 GiB of 6.0 GiB"));
        assert!(model_text(&loading, 1000).contains("mapped by  no program yet"));
    }

    #[test]
    fn the_process_lists_its_models()
    {
        let s = Status { pid: 4121, version: "3.1.0".into(), started_unix: 900, models: vec![model("a"), model("b")] };
        let t = status_text(&s, 1000 + 60);
        assert!(t.starts_with("dovata 3.1.0, pid 4121, up 2 min, 2 models, 9.7 GiB resident of 11.7 GiB\n"), "{t}");
        assert!(t.contains("\na, added 1 min ago\n") && t.contains("\nb, added 1 min ago\n"), "{t}");
        let u = usage_lines(&s);
        assert_eq!(u.lines().next(), Some("  dovata     pid 4121, 2 models, 9.7 GiB resident of 11.7 GiB"));
        assert!(u.contains("  model      a, 4.9 GiB resident of 5.9 GiB\n"), "{u}");
        let empty = Status { models: vec![], ..s };
        assert!(status_text(&empty, 1000).contains("No models. Add one with `dovata add <model>`."));
        assert!(usage_lines(&empty).contains("no models"));
    }

    #[test]
    fn spans_read_the_way_a_person_says_them()
    {
        assert_eq!(duration(45), "45 s");
        assert_eq!(duration(12 * 60 + 5), "12 min");
        assert_eq!(duration(3 * 3600 + 5 * 60), "3 h 5 min");
        assert_eq!(duration(2 * 86_400 + 4 * 3600), "2 d 4 h");
        assert_eq!((count(0).as_str(), count(1).as_str(), count(3).as_str()), ("no models", "1 model", "3 models"));
    }
}
