//! Author: Joe Ricks
//! File: src/cli/plan.rs
//! Description: The `dovata plan` command, which places every tensor of a model in memory
//! for one machine and prints where each one goes.

use std::path::{Path, PathBuf};

use dovata::hf_hub::{self, FetchError, FetchOptions, Fetched, Level, ModelSource, SelectError, Want};
use dovata::layout::{self, BindOptions, Layout};
use dovata::load::{KvType, Model};
use dovata::plan::{Backing, Plan, Role, Target};
use dovata::quantize::{self, Ftype, QuantError, QuantSpec};

use super::{json, out, parse_size, size, target_text, Fail};

/// The help text for `dovata plan`.
pub(crate) const HELP: &str = "\
usage: dovata plan <model> [options]

Place every tensor of a model in memory for one machine, and print where each one goes.

<model> is a GGUF file, or a Hugging Face link to a repo or to one file in it, such as
https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF

model:
  --quant TYPE         the quantization to plan, such as Q4_K_M. A repo's file of that
                       type is downloaded. When the repo has none, or the model is a local
                       file of another type, the unquantized model is quantized with
                       llama-quantize. A repo that offers several types needs this option.
  --allow-requantize   let --quant quantize a model that is already quantized
  --layout FILE        a layout file to use instead of dovata's built-in one

machine (default: this machine, as `dovata machine target` shows it):
  --target KIND        apple, cpu or cuda
  --memory SIZE        the memory the plan may use, such as 64G, to plan for another
                       machine; it needs --target
  --max-buffer SIZE    the largest GPU buffer; needed with --target apple --memory
  --page SIZE          the page size with --memory, default 16K for apple and 4K for cpu

step:
  --ctx N              KV cache length in tokens, default 4096
  --tokens N           tokens in one step, default 512
  --outputs N          tokens whose logits a step returns, default 1
  --kv TYPE            K and V cache type: f32, f16, bf16, q8_0, q5_1, q5_0, q4_1 or
                       q4_0, default f16
  --kv-k TYPE          the K cache type alone
  --kv-v TYPE          the V cache type alone

output:
  --json               print the whole plan as JSON, with every tensor's region and offset

Sizes use powers of 1024: 1G is 1,073,741,824 bytes.
";

/// Purpose: This enum lists the kinds of machine `--target` can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A Mac with Apple silicon.
    Apple,
    /// The CPU alone.
    Cpu,
    /// One NVIDIA GPU.
    Cuda,
}

/// Purpose: This struct holds the options of one `dovata plan` command line.
#[derive(Debug, Clone, PartialEq)]
struct Opts {
    /// The model, as written on the command line.
    model: String,
    /// `--quant`.
    quant: Option<Ftype>,
    /// `--allow-requantize`.
    allow_requantize: bool,
    /// `--layout`.
    layout: Option<PathBuf>,
    /// `--target`.
    kind: Option<Kind>,
    /// `--memory`.
    memory: Option<u64>,
    /// `--max-buffer`.
    max_buffer: Option<u64>,
    /// `--page`.
    page: Option<u64>,
    /// `--ctx`, `--tokens`, `--outputs`, `--kv`, `--kv-k` and `--kv-v`.
    step: BindOptions,
    /// `--json`.
    json: bool,
}

/// Purpose: This function runs `dovata plan`.
///
/// Inputs:
/// - `args`: the arguments after `plan`.
///
/// Outputs: `Ok(())`, or a [`Fail`] for a bad command line or a model that cannot be
/// planned.
pub(crate) fn main(
    args: &[String],
) -> Result<(), Fail>
{
    if args.is_empty() || args.iter().any(|a| matches!(a.as_str(), "help" | "--help" | "-h")) {
        out(HELP);
        return Ok(());
    }
    let opts = parse(args).map_err(|msg| Fail::Usage(format!("{msg}\n\nRun `dovata plan --help` for the options.")))?;
    run(&opts)
}

/// Purpose: This function reads the arguments of `dovata plan`.
///
/// Inputs:
/// - `args`: the arguments after `plan`, such as `["model.gguf", "--ctx", "8192"]`.
///
/// Outputs: the [`Opts`], or a message saying what is wrong with the arguments.
fn parse(
    args: &[String],
) -> Result<Opts, String>
{
    let mut o = Opts {
        model: String::new(),
        quant: None,
        allow_requantize: false,
        layout: None,
        kind: None,
        memory: None,
        max_buffer: None,
        page: None,
        step: BindOptions { n_ctx: 4096, n_tokens: 512, n_outputs: 1, kv_k: KvType::F16, kv_v: KvType::F16 },
        json: false,
    };
    let mut model = None;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = || args.get(i + 1).map(String::as_str).ok_or_else(|| format!("{flag} needs a value"));
        let count = |v: &str| v.parse::<u32>().ok().filter(|n| *n > 0).ok_or_else(|| format!("{flag} takes a whole number above 0, not {v:?}"));
        let kv = |v: &str| {
            KvType::parse(v).ok_or_else(|| {
                let names: Vec<&str> = KvType::ALL.iter().map(|k| k.name()).collect();
                format!("{flag} takes one of {}, not {v:?}", names.join(", "))
            })
        };
        let mut takes_value = true;
        match flag {
            "--quant" => o.quant = Some(Ftype::parse(value()?).map_err(|e| e.to_string())?),
            "--layout" => o.layout = Some(PathBuf::from(value()?)),
            "--target" => {
                o.kind = Some(match value()? {
                    "apple" => Kind::Apple,
                    "cpu" => Kind::Cpu,
                    "cuda" => Kind::Cuda,
                    v => return Err(format!("--target takes apple, cpu or cuda, not {v:?}")),
                })
            }
            "--memory" => o.memory = Some(parse_size(value()?)?),
            "--max-buffer" => o.max_buffer = Some(parse_size(value()?)?),
            "--page" => o.page = Some(parse_size(value()?).and_then(|p| page(p, flag))?),
            "--ctx" => o.step.n_ctx = count(value()?)?,
            "--tokens" => o.step.n_tokens = count(value()?)?,
            "--outputs" => o.step.n_outputs = count(value()?)?,
            "--kv" => {
                let t = kv(value()?)?;
                (o.step.kv_k, o.step.kv_v) = (t, t);
            }
            "--kv-k" => o.step.kv_k = kv(value()?)?,
            "--kv-v" => o.step.kv_v = kv(value()?)?,
            "--json" => (o.json, takes_value) = (true, false),
            "--allow-requantize" => (o.allow_requantize, takes_value) = (true, false),
            f if f.starts_with("--") => return Err(format!("unknown option {f:?}")),
            m => {
                if model.replace(m.to_owned()).is_some() {
                    return Err(format!("one model at a time; {m:?} is a second one"));
                }
                takes_value = false;
            }
        }
        i += if takes_value { 2 } else { 1 };
    }
    o.model = model.ok_or("name the model to plan: a GGUF file or a Hugging Face link")?;
    if o.step.n_outputs > o.step.n_tokens {
        return Err(format!("--outputs {} is more than --tokens {}", o.step.n_outputs, o.step.n_tokens));
    }
    if o.memory.is_none() && (o.max_buffer.is_some() || o.page.is_some()) {
        return Err("--max-buffer and --page describe another machine, so they need --memory".into());
    }
    if o.memory.is_some() && o.kind.is_none() {
        return Err("--memory describes another machine, so it needs --target apple, cpu or cuda".into());
    }
    if o.memory.is_some() && o.kind == Some(Kind::Apple) && o.max_buffer.is_none() {
        return Err("--target apple --memory needs --max-buffer, the largest Metal buffer of that Mac; \
                    `dovata machine target` on that Mac shows it"
            .into());
    }
    Ok(o)
}

/// Purpose: This function checks that a page size is a power of two.
///
/// Inputs:
/// - `p`: the page size in bytes.
/// - `flag`: the option it came from, for the message.
///
/// Outputs: the page size, or a message.
fn page(
    p: u64,
    flag: &str,
) -> Result<u64, String>
{
    if p.is_power_of_two() {
        Ok(p)
    } else {
        Err(format!("{flag} must be a power of two, such as 4K or 16K"))
    }
}

/// Purpose: This function gets the model, plans it and prints the plan.
///
/// Inputs:
/// - `o`: the command line's options.
///
/// Outputs: `Ok(())`, or a [`Fail`] saying why the model cannot be planned.
fn run(
    o: &Opts,
) -> Result<(), Fail>
{
    let target = target(o).map_err(Fail::Run)?;
    let parts = model_files(o)?;
    let model = Model::from_parts(&parts).map_err(|e| Fail::Run(format!("{}: {e}", parts[0].display())))?;
    let caller = match &o.layout {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| Fail::Run(format!("{}: {e}", path.display())))?;
            Some(Layout::from_toml(&text).map_err(|e| Fail::Run(format!("{}: {e}", path.display())))?)
        }
        None => None,
    };
    let layout = layout::for_model(&model, caller).map_err(|e| Fail::Run(e.to_string()))?;
    let plan = dovata::plan::plan(&model, &layout, &o.step, &target).map_err(|e| Fail::Run(e.to_string()))?;
    if o.json {
        out(&format!("{}\n", json(&plan)));
    } else {
        out(&summary(&model, &plan));
    }
    Ok(())
}

/// Purpose: This function builds the target machine from the command line, or reads this
/// machine when the command line describes none.
///
/// Inputs:
/// - `o`: the command line's options.
///
/// Outputs: the [`Target`], or a message saying why it cannot be built.
fn target(
    o: &Opts,
) -> Result<Target, String>
{
    if let (Some(memory), Some(kind)) = (o.memory, o.kind) {
        return Ok(match kind {
            Kind::Apple => Target::Apple {
                unified_bytes: memory,
                max_buffer_bytes: o.max_buffer.unwrap_or(memory),
                page_bytes: o.page.unwrap_or(16 << 10),
            },
            Kind::Cpu => Target::Cpu { ram_bytes: memory, page_bytes: o.page.unwrap_or(4 << 10) },
            Kind::Cuda => Target::Cuda { vram_bytes: memory },
        });
    }
    let m = dovata::machine::current().map_err(|e| format!("{e}. Describe the machine with --target and --memory instead"))?;
    match o.kind {
        None => Ok(m.target(false)),
        Some(Kind::Cpu) => Ok(m.target(true)),
        Some(Kind::Apple) => match m.target(false) {
            t @ Target::Apple { .. } => Ok(t),
            _ => Err("this machine has no Metal GPU; describe a Mac with --target apple --memory SIZE --max-buffer SIZE".into()),
        },
        Some(Kind::Cuda) => match m.nvidia.first() {
            Some(gpu) => Ok(Target::Cuda { vram_bytes: gpu.memory_total_bytes }),
            None => Err("this machine has no NVIDIA GPU; describe one with --target cuda --memory SIZE".into()),
        },
    }
}

/// Purpose: This function gets the model's files on disk: it downloads them for a Hub
/// link, and quantizes them when `--quant` asks for a type the source does not have.
///
/// Inputs:
/// - `o`: the command line's options.
///
/// Outputs: the files to plan, part 1 first, or a [`Fail`].
fn model_files(
    o: &Opts,
) -> Result<Vec<PathBuf>, Fail>
{
    let source = ModelSource::parse(&o.model).map_err(|e| Fail::Usage(e.to_string()))?;
    let fopts = FetchOptions::default();
    let run = |e: FetchError| Fail::Run(e.to_string());
    let fetched = match (&source, o.quant) {
        // A repo, and no type asked for: plan its only model, or stop and list them.
        (ModelSource::Hub { file: None, .. }, None) => {
            let offered = hf_hub::offered(&source, &fopts).map_err(run)?;
            match offered.as_slice() {
                [only] => fetch_level(&source, only, &fopts)?,
                [] => hf_hub::fetch(&source, Want::Unquantized, &fopts).map_err(run)?,
                several => return Err(Fail::Usage(choose(&source, several))),
            }
        }
        // A repo and a type: that type's file, or the unquantized model to quantize.
        (ModelSource::Hub { file: None, repo, .. }, Some(want)) => match hf_hub::fetch(&source, Want::Ftype(want), &fopts) {
            Ok(f) => f,
            Err(FetchError::Select { error: SelectError::NotOffered { .. }, .. }) => {
                eprintln!("dovata: {repo} has no {want} file, so dovata makes one from its unquantized model");
                hf_hub::fetch(&source, Want::Unquantized, &fopts).map_err(run)?
            }
            Err(e) => return Err(run(e)),
        },
        // A local file, or a link to one file.
        _ => hf_hub::fetch(&source, Want::Unquantized, &fopts).map_err(run)?,
    };
    match o.quant {
        Some(want) if fetched.ftype != Some(want) => quantized(&fetched, want, o.allow_requantize).map(|p| vec![p]),
        _ => Ok(fetched.parts),
    }
}

/// Purpose: This function downloads one model a Hub repo offers.
///
/// Inputs:
/// - `source`: the repo.
/// - `level`: the model, from [`hf_hub::offered`].
/// - `fopts`: the cache directory and the access token.
///
/// Outputs: the downloaded files, or a [`Fail`].
fn fetch_level(
    source: &ModelSource,
    level: &Level,
    fopts: &FetchOptions,
) -> Result<Fetched, Fail>
{
    let ModelSource::Hub { endpoint, repo, revision, .. } = source else {
        return Err(Fail::Run("a level comes from a Hub repo".into()));
    };
    let one = ModelSource::Hub { endpoint: endpoint.clone(), repo: repo.clone(), revision: revision.clone(), file: Some(level.files[0].clone()) };
    hf_hub::fetch(&one, Want::Unquantized, fopts).map_err(|e| Fail::Run(e.to_string()))
}

/// Purpose: This function writes the message that asks for `--quant` when a repo offers
/// several models.
///
/// Inputs:
/// - `source`: the repo.
/// - `offered`: its models, most precise first.
///
/// Outputs: the message.
fn choose(
    source: &ModelSource,
    offered: &[Level],
) -> String
{
    let mut s = format!("{source} offers {} models. Choose one with --quant, or link to its file:\n", offered.len());
    for l in offered {
        s += &format!("  {:<12} {}\n", l.label, l.files[0]);
    }
    s += "For a type the repo does not offer, dovata quantizes its unquantized model with llama-quantize.";
    s
}

/// Purpose: This function quantizes a model to one type, or reuses a copy made before.
///
/// Inputs:
/// - `fetched`: the model's files.
/// - `want`: the type to make.
/// - `allow_requantize`: whether an already quantized model may be quantized again.
///
/// Outputs: the quantized file, or a [`Fail`].
fn quantized(
    fetched: &Fetched,
    want: Ftype,
    allow_requantize: bool,
) -> Result<PathBuf, Fail>
{
    let input = &fetched.parts[0];
    let output = quantize::default_output(input, want);
    if declares(&output, want) {
        eprintln!("dovata: reusing {}, made from this model before", output.display());
        return Ok(output);
    }
    eprintln!("dovata: quantizing {} to {want} with llama-quantize, into {}", input.display(), output.display());
    let spec = QuantSpec { allow_requantize, ..QuantSpec::new(want) };
    match quantize::quantize(input, &output, &spec) {
        Ok(report) => Ok(report.output),
        Err(QuantError::AlreadyAt { input, .. }) => Ok(input),
        Err(e @ QuantError::Requantize { .. }) => Err(Fail::Run(format!("{e}. Add --allow-requantize to do it anyway"))),
        Err(e) => Err(Fail::Run(e.to_string())),
    }
}

/// Purpose: This function checks whether a file is a model of one type.
///
/// Inputs:
/// - `path`: the file.
/// - `ftype`: the type.
///
/// Outputs: `true` when the file opens as a model whose header declares `ftype`.
fn declares(
    path: &Path,
    ftype: Ftype,
) -> bool
{
    path.exists() && Model::open(path).is_ok_and(|m| m.meta().file_type() == Some(ftype.value()))
}

/// Purpose: This function formats a plan for a person: the model, the machine, the step
/// and each region.
///
/// Inputs:
/// - `model`: the model that was planned.
/// - `plan`: its plan.
///
/// Outputs: the text, one item per line.
fn summary(
    model: &Model,
    plan: &Plan,
) -> String
{
    let meta = model.meta();
    let ftype = meta.file_type().and_then(Ftype::from_file_type).map(|f| format!(", {f}")).unwrap_or_default();
    let mut s = format!("{}, {}{ftype}\n", meta.name().unwrap_or("unnamed model"), model.architecture());
    s += &format!("  machine    {}\n", target_text(&plan.target));
    if let Some(step) = &plan.step {
        let o = &step.options;
        s += &format!(
            "  step       {} tokens, {} with logits, a KV cache of {} tokens in {} for K and {} for V\n",
            o.n_tokens,
            o.n_outputs,
            o.n_ctx,
            o.kv_k.name(),
            o.kv_v.name()
        );
        s += &format!("  graph      {} ops over {} layers, {} tensors\n", step.ops.len(), step.n_layer, plan.entries.len());
    }
    s += "  regions\n";
    for r in &plan.regions {
        s += &format!("    {:<3} {:<8} {:>10}  {}\n", r.id, kind(r), size(r.bytes), holds(plan, r));
    }
    let allocated: u64 = plan.totals.allocated.values().sum();
    s += &format!("  total      {} mapped from the model file, {} allocated\n", size(plan.totals.mapped), size(allocated));
    for note in &plan.notes {
        s += &format!("  note       {note}\n");
    }
    s += &format!("  table hash {}\n", plan.table_hash);
    s += "Add --json for every tensor's region and offset.\n";
    s
}

/// Purpose: This function names the kind of memory a region is.
///
/// Inputs:
/// - `r`: the region.
///
/// Outputs: a name such as `mmap` or `unified`.
fn kind(
    r: &dovata::plan::Region,
) -> &'static str
{
    use dovata::plan::RegionKind;
    match r.kind {
        RegionKind::Mmap => "mmap",
        RegionKind::Host => "host",
        RegionKind::Vram => "vram",
        RegionKind::Unified => "unified",
    }
}

/// Purpose: This function says what a region holds.
///
/// Inputs:
/// - `plan`: the plan.
/// - `r`: one of its regions.
///
/// Outputs: a phrase such as `weights, part 1 from byte 0` or `KV cache`.
fn holds(
    plan: &Plan,
    r: &dovata::plan::Region,
) -> String
{
    if let Backing::File { part, file_start } = r.backing {
        return format!("weights, part {} from byte {file_start}", part + 1);
    }
    let mut what: Vec<&str> = Vec::new();
    for e in plan.entries.iter().filter(|e| e.region == r.id) {
        let w = match e.role {
            Role::Weight => "weights",
            Role::Kv => "KV cache",
            Role::State => "recurrent state",
            Role::Input | Role::Scratch | Role::Output => "step inputs, scratch and outputs",
            Role::View { .. } => continue,
        };
        if !what.contains(&w) {
            what.push(w);
        }
    }
    what.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Purpose: This function reads a `dovata plan` command line written as one string.
    ///
    /// Inputs:
    /// - `line`: the arguments after `plan`, separated by spaces.
    ///
    /// Outputs: what [`parse`] returns for them.
    fn p(
        line: &str,
    ) -> Result<Opts, String>
    {
        parse(&line.split_whitespace().map(str::to_owned).collect::<Vec<_>>())
    }

    #[test]
    fn a_model_alone_takes_the_defaults()
    {
        let o = p("m.gguf").unwrap();
        assert_eq!(o.model, "m.gguf");
        assert_eq!((o.step.n_ctx, o.step.n_tokens, o.step.n_outputs), (4096, 512, 1));
        assert_eq!((o.step.kv_k, o.step.kv_v), (KvType::F16, KvType::F16));
        assert_eq!((o.kind, o.memory, o.quant, o.json), (None, None, None, false));
    }

    #[test]
    fn options_parse()
    {
        let o = p("m.gguf --quant q4_k_m --target apple --memory 64G --max-buffer 48G --ctx 32768 --kv q8_0 --kv-v q4_0 --json").unwrap();
        assert_eq!(o.quant, Some(Ftype::Q4_K_M));
        assert_eq!(o.kind, Some(Kind::Apple));
        assert_eq!((o.memory, o.max_buffer), (Some(64 << 30), Some(48 << 30)));
        assert_eq!(o.step.n_ctx, 32768);
        assert_eq!((o.step.kv_k, o.step.kv_v), (KvType::Q8_0, KvType::Q4_0));
        assert!(o.json);
        let t = target(&o).unwrap();
        assert_eq!(t, Target::Apple { unified_bytes: 64 << 30, max_buffer_bytes: 48 << 30, page_bytes: 16384 });
        let cpu = p("m.gguf --target cpu --memory 128G --page 64K").unwrap();
        assert_eq!(target(&cpu).unwrap(), Target::Cpu { ram_bytes: 128 << 30, page_bytes: 65536 });
    }

    #[test]
    fn bad_command_lines_are_refused()
    {
        for bad in [
            "",
            "--json",
            "a.gguf b.gguf",
            "m.gguf --quant Q9",
            "m.gguf --target tpu",
            "m.gguf --memory 64G",
            "m.gguf --max-buffer 8G",
            "m.gguf --target apple --memory 64G",
            "m.gguf --ctx 0",
            "m.gguf --tokens 4 --outputs 8",
            "m.gguf --kv q2_k",
            "m.gguf --page 3K --memory 1G --target cpu",
            "m.gguf --frobnicate",
        ] {
            assert!(p(bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn several_models_are_listed_for_a_choice()
    {
        let source = ModelSource::parse("https://huggingface.co/org/repo").unwrap();
        let level = |label: &str| Level { label: label.into(), ftype: Ftype::parse(label).ok(), files: vec![format!("m-{label}.gguf")] };
        let msg = choose(&source, &[level("BF16"), level("Q8_0"), level("Q4_K_M")]);
        assert!(msg.contains("offers 3 models") && msg.contains("--quant"), "{msg}");
        assert!(msg.contains("m-Q4_K_M.gguf"), "{msg}");
    }
}
