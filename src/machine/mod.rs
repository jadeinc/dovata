//! Author: Joe Ricks
//! File: src/machine/mod.rs
//! Description: Describes the current machine's memory and GPUs, and reads the output of
//! the system tools that report them.

use std::fmt;

use serde::Serialize;

use crate::plan::Target;

#[cfg(feature = "machine")]
mod probe;
#[cfg(all(feature = "machine", target_os = "macos"))]
mod metal;

#[cfg(feature = "machine")]
pub use probe::{current, usage};

/// Purpose: This function returns `NotBuilt` in a build without the `machine` feature.
///
/// Inputs: none.
///
/// Outputs: always [`MachineError::NotBuilt`].
#[cfg(not(feature = "machine"))]
pub fn current() -> Result<Machine, MachineError> {
    Err(MachineError::NotBuilt)
}

/// Purpose: This function returns `NotBuilt` in a build without the `machine` feature.
///
/// Inputs:
/// - `process`: a program name to report on; unused.
///
/// Outputs: always [`MachineError::NotBuilt`].
#[cfg(not(feature = "machine"))]
pub fn usage(process: Option<&str>) -> Result<Usage, MachineError> {
    let _ = process;
    Err(MachineError::NotBuilt)
}

/// Purpose: This struct holds what the current machine has: its memory, its page size and
/// its GPUs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Machine {
    /// The operating system and its version, such as `macOS 26.5.2`.
    pub os: String,
    /// The processor, such as `Apple M4`, when the system names it.
    pub chip: Option<String>,
    /// Physical memory, in bytes. On Apple silicon the GPU shares it.
    pub memory_bytes: u64,
    /// The memory page size, in bytes.
    pub page_bytes: u64,
    /// Metal's limits, on a Mac with a Metal GPU.
    pub metal: Option<MetalLimits>,
    /// Every NVIDIA GPU `nvidia-smi` reports, in its order.
    pub nvidia: Vec<Gpu>,
}

/// Purpose: This struct holds the two memory limits Metal reports for a Mac's GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MetalLimits {
    /// `recommendedMaxWorkingSetSize`: how much memory the GPU can use before
    /// performance suffers, in bytes.
    pub working_set_bytes: u64,
    /// `maxBufferLength`: the largest single Metal buffer, in bytes.
    pub max_buffer_bytes: u64,
    /// `hasUnifiedMemory`: the GPU shares the CPU's memory.
    pub unified: bool,
}

/// Purpose: This struct holds one NVIDIA GPU's memory, load, temperature and clocks, as
/// `nvidia-smi` reports them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Gpu {
    /// The GPU's index, as `nvidia-smi` and CUDA number it.
    pub index: u32,
    /// The product name, such as `NVIDIA GeForce RTX 4090`.
    pub name: String,
    /// Total VRAM, in bytes.
    pub memory_total_bytes: u64,
    /// VRAM in use, in bytes.
    pub memory_used_bytes: Option<u64>,
    /// VRAM free, in bytes.
    pub memory_free_bytes: Option<u64>,
    /// How busy the GPU was over the last sample period, from 0 to 100.
    pub load_percent: Option<u32>,
    /// The GPU's temperature, in degrees Celsius.
    pub temperature_c: Option<u32>,
    /// The shader clock, in MHz.
    pub sm_clock_mhz: Option<u32>,
    /// The memory clock, in MHz.
    pub memory_clock_mhz: Option<u32>,
}

/// Purpose: This struct holds how memory is being used right now.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Usage {
    /// The machine's main memory.
    pub memory: MemoryUse,
    /// Every NVIDIA GPU, with its current memory, load and temperature.
    pub nvidia: Vec<Gpu>,
    /// The running programs that match the name asked for, if any was.
    pub processes: Vec<ProcessUse>,
}

/// Purpose: This struct holds a snapshot of the machine's main memory.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemoryUse {
    /// Physical memory, in bytes.
    pub total_bytes: u64,
    /// Memory programs hold and the system cannot hand out, in bytes. On macOS this is
    /// app memory plus immovable plus compressed; on Linux it is total less `MemAvailable`.
    pub used_bytes: u64,
    /// Memory a program could still get, total less used, in bytes.
    pub available_bytes: u64,
    /// Memory the system cannot move, compress or swap out, which `vm_stat` calls wired,
    /// in bytes. macOS only.
    pub immovable_bytes: Option<u64>,
    /// Memory the system's compressor holds, in bytes. macOS reports it; Linux reports it
    /// when zswap is on.
    pub compressed_bytes: Option<u64>,
    /// File data kept in memory that the system can drop when it needs room, in bytes.
    pub cached_bytes: Option<u64>,
    /// Swap space, in bytes.
    pub swap_total_bytes: u64,
    /// Swap in use, in bytes.
    pub swap_used_bytes: u64,
    /// How hard the system is working to find memory.
    pub pressure: Option<Pressure>,
}

/// Purpose: This enum holds a memory pressure reading, in the form each operating system
/// reports it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Pressure {
    /// macOS `memory_pressure`: the share of memory free, from 0 to 100. Lower is worse.
    FreePercent(u8),
    /// Linux pressure stall information: the share of the last 10 seconds in which some
    /// task waited for memory, from 0 to 100. Higher is worse.
    StallPercent(f64),
}

/// Purpose: This struct holds one running program's memory use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProcessUse {
    /// The process id.
    pub pid: u32,
    /// The command, as `ps` shows it.
    pub command: String,
    /// Resident memory: the bytes of the program held in physical memory.
    pub rss_bytes: u64,
}

/// Purpose: This enum lists the errors reading the machine can return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MachineError {
    /// This build of dovata cannot read the machine: the `machine` feature is off.
    NotBuilt,
    /// dovata does not know how to read this operating system.
    Unsupported(String),
    /// A system tool could not be run, or failed.
    Command { program: String, detail: String },
    /// A system tool's output was not in the expected form.
    Parse { what: String, detail: String },
}

impl fmt::Display for MachineError {
    /// Purpose: This function formats errors from reading the machine.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MachineError::NotBuilt => write!(
                f,
                "this build of dovata cannot read the machine: it was compiled without the `machine` cargo feature"
            ),
            MachineError::Unsupported(os) => write!(f, "dovata reads memory on macOS and Linux, not on {os}"),
            MachineError::Command { program, detail } => write!(f, "{program} failed: {detail}"),
            MachineError::Parse { what, detail } => write!(f, "cannot read {what}: {detail}"),
        }
    }
}

impl std::error::Error for MachineError {}

impl Machine {
    /// Purpose: This function describes this machine as a target for the planner.
    ///
    /// Inputs:
    /// - `cpu_only`: describe the CPU even when the machine has a GPU.
    ///
    /// Outputs: an Apple target on a Mac with Metal, using Metal's working set and buffer
    /// limits; else a CUDA target for the first NVIDIA GPU, using its total VRAM; else a
    /// CPU target with all physical memory.
    pub fn target(&self, cpu_only: bool) -> Target {
        if !cpu_only {
            if let Some(m) = self.metal {
                return Target::Apple {
                    unified_bytes: m.working_set_bytes,
                    max_buffer_bytes: m.max_buffer_bytes,
                    page_bytes: self.page_bytes,
                };
            }
            if let Some(gpu) = self.nvidia.first() {
                return Target::Cuda { vram_bytes: gpu.memory_total_bytes };
            }
        }
        Target::Cpu { ram_bytes: self.memory_bytes, page_bytes: self.page_bytes }
    }
}

/// Purpose: This function reads macOS `vm_stat` output into a memory snapshot.
///
/// Inputs:
/// - `text`: what `vm_stat` printed.
/// - `total_bytes`: physical memory, from `sysctl hw.memsize`.
///
/// Outputs: the snapshot without swap or pressure, or a [`MachineError::Parse`].
pub fn parse_vm_stat(text: &str, total_bytes: u64) -> Result<MemoryUse, MachineError> {
    let fail = |detail: &str| MachineError::Parse { what: "vm_stat output".into(), detail: detail.into() };
    let page: u64 = text
        .lines()
        .next()
        .and_then(|l| l.split("page size of ").nth(1))
        .and_then(|r| r.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| fail("no page size on its first line"))?;
    let pages = |label: &str| -> Result<u64, MachineError> {
        text.lines()
            .find_map(|l| l.strip_prefix(label)?.strip_prefix(':'))
            .and_then(|v| v.trim().trim_end_matches('.').parse().ok())
            .ok_or_else(|| fail(&format!("no \"{label}\" line")))
    };
    let app = pages("Anonymous pages")?.saturating_sub(pages("Pages purgeable")?);
    let immovable = pages("Pages wired down")? * page;
    let compressed = pages("Pages occupied by compressor")? * page;
    let used = (app * page + immovable + compressed).min(total_bytes);
    Ok(MemoryUse {
        total_bytes,
        used_bytes: used,
        available_bytes: total_bytes - used,
        immovable_bytes: Some(immovable),
        compressed_bytes: Some(compressed),
        cached_bytes: Some(pages("File-backed pages")? * page),
        swap_total_bytes: 0,
        swap_used_bytes: 0,
        pressure: None,
    })
}

/// Purpose: This function reads macOS `sysctl vm.swapusage` output.
///
/// Inputs:
/// - `text`: the output, such as `total = 2048.00M  used = 512.50M  free = 1535.50M`.
///
/// Outputs: total and used swap in bytes, or a [`MachineError::Parse`].
pub fn parse_swapusage(text: &str) -> Result<(u64, u64), MachineError> {
    let field = |name: &str| -> Option<u64> {
        let rest = text.split(&format!("{name} = ")).nth(1)?;
        let value = rest.split_whitespace().next()?;
        let (number, unit) = value.split_at(value.find(|c: char| c.is_ascii_alphabetic())?);
        let scale = match unit {
            "K" => 1u64 << 10,
            "M" => 1 << 20,
            "G" => 1 << 30,
            "T" => 1 << 40,
            _ => return None,
        };
        let number: f64 = number.parse().ok()?;
        Some((number * scale as f64).round() as u64)
    };
    match (field("total"), field("used")) {
        (Some(total), Some(used)) => Ok((total, used)),
        _ => Err(MachineError::Parse { what: "vm.swapusage".into(), detail: format!("unexpected text {:?}", text.trim()) }),
    }
}

/// Purpose: This function reads macOS `memory_pressure -Q` output.
///
/// Inputs:
/// - `text`: the output, whose last line is `System-wide memory free percentage: 54%`.
///
/// Outputs: the free percentage, or `None` when the line is missing.
pub fn parse_memory_pressure(text: &str) -> Option<Pressure> {
    let line = text.lines().find(|l| l.contains("memory free percentage"))?;
    let value = line.rsplit(':').next()?.trim().trim_end_matches('%');
    value.parse().ok().map(Pressure::FreePercent)
}

/// Purpose: This function reads Linux `/proc/meminfo` into a memory snapshot.
///
/// Inputs:
/// - `text`: the file's contents.
///
/// Outputs: the snapshot without pressure, or a [`MachineError::Parse`].
pub fn parse_meminfo(text: &str) -> Result<MemoryUse, MachineError> {
    let kib = |key: &str| -> Option<u64> {
        let line = text.lines().find(|l| l.split(':').next() == Some(key))?;
        let value: u64 = line.split(':').nth(1)?.split_whitespace().next()?.parse().ok()?;
        Some(value * 1024)
    };
    let need = |key: &str| {
        kib(key).ok_or_else(|| MachineError::Parse { what: "/proc/meminfo".into(), detail: format!("no {key} line") })
    };
    let total = need("MemTotal")?;
    let available = need("MemAvailable")?.min(total);
    let swap_total = need("SwapTotal")?;
    let swap_free = need("SwapFree")?.min(swap_total);
    Ok(MemoryUse {
        total_bytes: total,
        used_bytes: total - available,
        available_bytes: available,
        immovable_bytes: None,
        compressed_bytes: kib("Zswap").filter(|z| *z > 0),
        cached_bytes: kib("Cached"),
        swap_total_bytes: swap_total,
        swap_used_bytes: swap_total - swap_free,
        pressure: None,
    })
}

/// Purpose: This function reads Linux `/proc/pressure/memory`.
///
/// Inputs:
/// - `text`: the file's contents, whose first line is `some avg10=1.25 avg60=...`.
///
/// Outputs: the `some avg10` stall percentage, or `None` when it is missing.
pub fn parse_psi(text: &str) -> Option<Pressure> {
    let line = text.lines().find(|l| l.starts_with("some "))?;
    let value = line.split_whitespace().find_map(|w| w.strip_prefix("avg10="))?;
    value.parse().ok().map(Pressure::StallPercent)
}

/// The fields dovata asks `nvidia-smi --query-gpu` for, in order.
pub const NVIDIA_QUERY: &str =
    "index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem";

/// Purpose: This function reads `nvidia-smi` output for the fields in [`NVIDIA_QUERY`].
///
/// Inputs:
/// - `text`: the output of `nvidia-smi --query-gpu=<NVIDIA_QUERY> --format=csv,noheader,nounits`.
///
/// Outputs: one [`Gpu`] per line, or a [`MachineError::Parse`].
pub fn parse_nvidia_smi(text: &str) -> Result<Vec<Gpu>, MachineError> {
    let fields = NVIDIA_QUERY.split(',').count();
    let mut gpus = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        if cols.len() != fields {
            return Err(MachineError::Parse {
                what: "nvidia-smi output".into(),
                detail: format!("expected {fields} fields, found {}: {line:?}", cols.len()),
            });
        }
        // nvidia-smi prints `[N/A]` or `[Not Supported]` for a value it cannot read.
        let number = |i: usize| cols[i].parse::<u64>().ok();
        let mib = |i: usize| number(i).map(|v| v * 1024 * 1024);
        let small = |i: usize| number(i).and_then(|v| u32::try_from(v).ok());
        gpus.push(Gpu {
            index: small(0).ok_or_else(|| MachineError::Parse { what: "nvidia-smi output".into(), detail: format!("bad index in {line:?}") })?,
            name: cols[1].to_owned(),
            memory_total_bytes: mib(2).unwrap_or(0),
            memory_used_bytes: mib(3),
            memory_free_bytes: mib(4),
            load_percent: small(5),
            temperature_c: small(6),
            sm_clock_mhz: small(7),
            memory_clock_mhz: small(8),
        });
    }
    Ok(gpus)
}

/// Purpose: This function finds the running programs with a given name in `ps` output.
///
/// Inputs:
/// - `text`: the output of `ps -A -o pid=,rss=,comm=`, one process per line.
/// - `name`: the program name to match against the command's last path part, such as
///   `qwen-chat`, or the whole command.
///
/// Outputs: every matching process, in the order `ps` listed them.
pub fn parse_ps(text: &str, name: &str) -> Vec<ProcessUse> {
    text.lines()
        .filter_map(|line| {
            let mut words = line.trim_start().splitn(2, char::is_whitespace);
            let pid = words.next()?.parse().ok()?;
            let rest = words.next()?.trim_start();
            let (rss, command) = rest.split_once(char::is_whitespace)?;
            let command = command.trim();
            let base = command.rsplit('/').next().unwrap_or(command);
            (base == name || command == name)
                .then(|| Some(ProcessUse { pid, command: command.to_owned(), rss_bytes: rss.parse::<u64>().ok()? * 1024 }))
                .flatten()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn vm_stat_gives_used_immovable_and_compressed() {
        let text = include_str!("testdata/macos-vm_stat.txt");
        let m = parse_vm_stat(text, 24 * GIB).unwrap();
        let page = 16384;
        let field = |label: &str| -> u64 {
            text.lines().find_map(|l| l.strip_prefix(label)?.strip_prefix(':')).unwrap().trim().trim_end_matches('.').parse().unwrap()
        };
        let app = field("Anonymous pages") - field("Pages purgeable");
        let expect = (app + field("Pages wired down") + field("Pages occupied by compressor")) * page;
        assert_eq!(m.used_bytes, expect.min(24 * GIB));
        assert_eq!(m.used_bytes + m.available_bytes, m.total_bytes);
        assert_eq!(m.immovable_bytes, Some(field("Pages wired down") * page));
        assert_eq!(m.cached_bytes, Some(field("File-backed pages") * page));
        assert!(parse_vm_stat("Mach Virtual Memory Statistics:\n", GIB).is_err(), "no page size");
    }

    #[test]
    fn swapusage_reads_each_unit() {
        assert_eq!(parse_swapusage(include_str!("testdata/macos-swapusage.txt")), Ok((0, 0)));
        assert_eq!(parse_swapusage("total = 2048.00M  used = 512.50M  free = 1535.50M  (encrypted)"), Ok((2 * GIB, 537_395_200)));
        assert_eq!(parse_swapusage("total = 1.50G  used = 0.00M  free = 1.50G"), Ok((1_610_612_736, 0)));
        assert!(parse_swapusage("nonsense").is_err());
    }

    #[test]
    fn memory_pressure_reads_the_free_percentage() {
        let p = parse_memory_pressure(include_str!("testdata/macos-memory_pressure.txt"));
        assert!(matches!(p, Some(Pressure::FreePercent(v)) if v <= 100), "{p:?}");
        assert_eq!(parse_memory_pressure("System-wide memory free percentage: 7%"), Some(Pressure::FreePercent(7)));
        assert_eq!(parse_memory_pressure("nothing here"), None);
    }

    #[test]
    fn meminfo_gives_used_from_available() {
        let m = parse_meminfo(include_str!("testdata/linux-meminfo.txt")).unwrap();
        assert_eq!(m.total_bytes, 65_842_148 * 1024);
        assert_eq!(m.available_bytes, 41_207_716 * 1024);
        assert_eq!(m.used_bytes, (65_842_148 - 41_207_716) * 1024);
        assert_eq!(m.cached_bytes, Some(36_148_120 * 1024));
        assert_eq!(m.compressed_bytes, None, "zswap is off");
        assert_eq!((m.swap_total_bytes, m.swap_used_bytes), (8_388_604 * 1024, (8_388_604 - 8_120_060) * 1024));
        assert!(parse_meminfo("MemTotal: 1 kB\n").is_err());
    }

    #[test]
    fn psi_reads_some_avg10() {
        assert_eq!(parse_psi(include_str!("testdata/linux-pressure-memory.txt")), Some(Pressure::StallPercent(1.25)));
        assert_eq!(parse_psi(""), None);
    }

    #[test]
    fn nvidia_smi_reads_each_gpu() {
        let gpus = parse_nvidia_smi(include_str!("testdata/nvidia-smi.csv")).unwrap();
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 4090");
        assert_eq!(gpus[0].memory_total_bytes, 24_564 << 20);
        assert_eq!(gpus[0].memory_used_bytes, Some(18_211 << 20));
        assert_eq!((gpus[0].load_percent, gpus[0].temperature_c), (Some(97), Some(71)));
        assert_eq!(gpus[1].sm_clock_mhz, None, "[N/A] is no value");
        assert_eq!(gpus[1].memory_clock_mhz, Some(1593));
        assert!(parse_nvidia_smi("0, short line").is_err());
        assert_eq!(parse_nvidia_smi("\n"), Ok(vec![]));
    }

    #[test]
    fn ps_matches_by_program_name() {
        let text = include_str!("testdata/macos-ps.txt");
        let found = parse_ps(text, "qwen-chat");
        assert_eq!(found.iter().map(|p| p.pid).collect::<Vec<_>>(), [5120, 5121]);
        assert_eq!(found[0].rss_bytes, 2_411_520 * 1024);
        let code = parse_ps(text, "Code Helper (Renderer)");
        assert_eq!(code.len(), 1, "a name with spaces");
        assert_eq!(parse_ps(text, "/sbin/launchd")[0].pid, 1, "the whole command matches too");
        assert!(parse_ps(text, "chat").is_empty(), "a partial name does not");
    }

    #[test]
    fn the_target_follows_the_hardware() {
        let mut m = Machine {
            os: "macOS 26.5.2".into(),
            chip: Some("Apple M4".into()),
            memory_bytes: 24 * GIB,
            page_bytes: 16384,
            metal: Some(MetalLimits { working_set_bytes: 19_069_665_280, max_buffer_bytes: 14_302_248_960, unified: true }),
            nvidia: vec![],
        };
        assert_eq!(
            m.target(false),
            Target::Apple { unified_bytes: 19_069_665_280, max_buffer_bytes: 14_302_248_960, page_bytes: 16384 }
        );
        assert_eq!(m.target(true), Target::Cpu { ram_bytes: 24 * GIB, page_bytes: 16384 });
        m.metal = None;
        m.nvidia = parse_nvidia_smi(include_str!("testdata/nvidia-smi.csv")).unwrap();
        assert_eq!(m.target(false), Target::Cuda { vram_bytes: 24_564 << 20 }, "the first GPU only");
    }

    #[test]
    fn pressure_serializes_in_snake_case() {
        assert_eq!(serde_json::to_string(&Pressure::FreePercent(55)).unwrap(), r#"{"free_percent":55}"#);
    }

    #[test]
    #[cfg(not(feature = "machine"))]
    fn reading_the_machine_is_refused_without_feature() {
        assert_eq!(current(), Err(MachineError::NotBuilt));
        assert_eq!(usage(None).unwrap_err(), MachineError::NotBuilt);
        assert!(MachineError::NotBuilt.to_string().contains("`machine` cargo feature"));
    }
}
