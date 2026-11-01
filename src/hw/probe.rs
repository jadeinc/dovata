//! Author: Joe Ricks
//! File: src/hw/probe.rs
//! Description: Reads the current machine's memory and GPUs by running its system tools.

use std::path::PathBuf;
use std::process::Command;

use super::{
    parse_memory_pressure, parse_meminfo, parse_nvidia_smi, parse_ps, parse_psi, parse_swapusage, parse_vm_stat, Gpu,
    HwError, Machine, MetalLimits, Usage, NVIDIA_QUERY,
};

/// Purpose: This function reads what the current machine has: its memory, page size and
/// GPUs.
///
/// Inputs: none.
///
/// Outputs: the [`Machine`], or a [`HwError`] when a system tool fails or the operating
/// system is not macOS or Linux.
pub fn machine() -> Result<Machine, HwError> {
    if cfg!(target_os = "macos") {
        Ok(Machine {
            os: format!("macOS {}", run("/usr/bin/sw_vers", &["-productVersion"])?.trim()),
            chip: run("/usr/sbin/sysctl", &["-n", "machdep.cpu.brand_string"]).ok().map(|s| s.trim().to_owned()),
            memory_bytes: sysctl_number("hw.memsize")?,
            page_bytes: sysctl_number("hw.pagesize")?,
            metal: metal_limits(),
            nvidia: nvidia_gpus()?,
        })
    } else if cfg!(target_os = "linux") {
        let meminfo = read("/proc/meminfo")?;
        let chip = read("/proc/cpuinfo").ok().and_then(|c| {
            c.lines().find_map(|l| Some(l.strip_prefix("model name")?.split_once(':')?.1.trim().to_owned()))
        });
        let page = run("getconf", &["PAGESIZE"])?;
        Ok(Machine {
            os: run("uname", &["-sr"])?.trim().to_owned(),
            chip,
            memory_bytes: parse_meminfo(&meminfo)?.total_bytes,
            page_bytes: page.trim().parse().map_err(|_| parse_error("getconf PAGESIZE", &page))?,
            metal: None,
            nvidia: nvidia_gpus()?,
        })
    } else {
        Err(HwError::Unsupported(std::env::consts::OS.to_owned()))
    }
}

/// Purpose: This function reads how the current machine's memory and GPUs are being used
/// right now.
///
/// Inputs:
/// - `process`: a program name, such as `qwen-chat`, whose running copies to report on.
///
/// Outputs: the [`Usage`], or a [`HwError`] when a system tool fails or the operating
/// system is not macOS or Linux.
pub fn usage(process: Option<&str>) -> Result<Usage, HwError> {
    let memory = if cfg!(target_os = "macos") {
        let mut m = parse_vm_stat(&run("/usr/bin/vm_stat", &[])?, sysctl_number("hw.memsize")?)?;
        (m.swap_total_bytes, m.swap_used_bytes) = parse_swapusage(&run("/usr/sbin/sysctl", &["-n", "vm.swapusage"])?)?;
        m.pressure = run("/usr/bin/memory_pressure", &["-Q"]).ok().and_then(|t| parse_memory_pressure(&t));
        m
    } else if cfg!(target_os = "linux") {
        let mut m = parse_meminfo(&read("/proc/meminfo")?)?;
        m.pressure = read("/proc/pressure/memory").ok().and_then(|t| parse_psi(&t));
        m
    } else {
        return Err(HwError::Unsupported(std::env::consts::OS.to_owned()));
    };
    let processes = match process {
        Some(name) => {
            let ps = if cfg!(target_os = "macos") { "/bin/ps" } else { "ps" };
            parse_ps(&run(ps, &["-A", "-o", "pid=,rss=,comm="])?, name)
        }
        None => Vec::new(),
    };
    Ok(Usage { memory, nvidia: nvidia_gpus()?, processes })
}

/// Purpose: This function reads every NVIDIA GPU through `nvidia-smi`.
///
/// Inputs: none.
///
/// Outputs: one [`Gpu`] per GPU; none when `nvidia-smi` is not on `PATH`; or a
/// [`HwError`] when it is installed but fails, as it does when the driver is not loaded.
fn nvidia_gpus() -> Result<Vec<Gpu>, HwError> {
    let Some(program) = find_on_path("nvidia-smi") else {
        return Ok(Vec::new());
    };
    let query = format!("--query-gpu={NVIDIA_QUERY}");
    parse_nvidia_smi(&run(&program.to_string_lossy(), &[&query, "--format=csv,noheader,nounits"])?)
}

/// Purpose: This function reads the Mac GPU's memory limits from Metal.
///
/// Inputs: none.
///
/// Outputs: the limits, or `None` on a machine without a Metal GPU.
#[cfg(target_os = "macos")]
fn metal_limits() -> Option<MetalLimits> {
    super::metal::limits()
}

/// Purpose: This function stands in for the Metal reader on systems without Metal.
///
/// Inputs: none.
///
/// Outputs: always `None`.
#[cfg(not(target_os = "macos"))]
fn metal_limits() -> Option<MetalLimits> {
    None
}

/// Purpose: This function reads one numeric value with macOS `sysctl`.
///
/// Inputs:
/// - `name`: the value's name, such as `hw.memsize`.
///
/// Outputs: the number, or a [`HwError`].
fn sysctl_number(name: &str) -> Result<u64, HwError> {
    let text = run("/usr/sbin/sysctl", &["-n", name])?;
    text.trim().parse().map_err(|_| parse_error(&format!("sysctl {name}"), &text))
}

/// Purpose: This function runs a system tool and returns what it printed.
///
/// Inputs:
/// - `program`: the tool's path, or its name to find on `PATH`.
/// - `args`: its arguments.
///
/// Outputs: its standard output, or a [`HwError::Command`] when it cannot start or exits
/// with an error.
fn run(program: &str, args: &[&str]) -> Result<String, HwError> {
    let fail = |detail: String| HwError::Command { program: program.to_owned(), detail };
    let out = Command::new(program).args(args).output().map_err(|e| fail(format!("cannot start it: {e}")))?;
    if !out.status.success() {
        return Err(fail(format!("{}: {}", out.status, String::from_utf8_lossy(&out.stderr).trim())));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Purpose: This function reads a system file, such as `/proc/meminfo`.
///
/// Inputs:
/// - `path`: the file's path.
///
/// Outputs: its contents, or a [`HwError::Command`] naming the file.
fn read(path: &str) -> Result<String, HwError> {
    std::fs::read_to_string(path).map_err(|e| HwError::Command { program: path.to_owned(), detail: e.to_string() })
}

/// Purpose: This function finds a program on `PATH`.
///
/// Inputs:
/// - `name`: the program's name.
///
/// Outputs: its path, or `None` when it is not installed.
fn find_on_path(name: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).map(|dir| dir.join(name)).find(|p| p.is_file())
}

/// Purpose: This function builds the error for a tool's output that is not a number.
///
/// Inputs:
/// - `what`: the tool and value, such as `sysctl hw.memsize`.
/// - `text`: what it printed.
///
/// Outputs: the [`HwError::Parse`].
fn parse_error(what: &str, text: &str) -> HwError {
    HwError::Parse { what: what.to_owned(), detail: format!("expected a number, found {:?}", text.trim()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn this_machine_reads() {
        let m = machine().unwrap();
        assert!(m.memory_bytes > 1 << 30, "{m:?}");
        assert!(m.page_bytes.is_power_of_two(), "{m:?}");
        if let Some(metal) = m.metal {
            assert!(metal.working_set_bytes <= m.memory_bytes && metal.max_buffer_bytes <= metal.working_set_bytes, "{m:?}");
        }
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert!(m.metal.is_some_and(|l| l.unified), "Apple silicon has a unified Metal GPU: {m:?}");
        }
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn this_machines_use_reads_and_finds_this_test() {
        let me = std::env::current_exe().unwrap();
        let name = me.file_name().unwrap().to_string_lossy().into_owned();
        let u = usage(Some(&name)).unwrap();
        assert!(u.memory.used_bytes > 0 && u.memory.used_bytes <= u.memory.total_bytes, "{u:?}");
        assert_eq!(u.memory.used_bytes + u.memory.available_bytes, u.memory.total_bytes);
        assert!(u.processes.iter().any(|p| p.pid == std::process::id() && p.rss_bytes > 0), "{name}: {:?}", u.processes);
    }
}
