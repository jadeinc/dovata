//! Author: Joe Ricks
//! File: src/jade-gpu-mon/src/main.rs
//! Description: Shows a live terminal dashboard of one NVIDIA GPU's memory and usage.

use std::io::{stdout, Write};
use std::time::Duration;

use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{poll, read, Event, KeyCode, KeyModifiers},
    execute, queue,
    style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::{
        disable_raw_mode, enable_raw_mode, size, Clear, ClearType, EnterAlternateScreen,
        LeaveAlternateScreen,
    },
};
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::Nvml;

const POLL_MS: u64 = 500;

/// Purpose: This function draws a text bar showing a percentage.
///
/// Inputs:
/// - `pct`: the percentage, 0 to 100.
/// - `width`: the bar's width in columns.
///
/// Outputs: the bar as text.
fn bar(pct: u32, width: usize) -> String {
    let n = ((pct as usize) * width / 100).min(width);
    format!("{}{}", "█".repeat(n), "░".repeat(width - n))
}

/// Purpose: This function picks a bar's color from its percentage.
///
/// Inputs:
/// - `pct`: the percentage.
///
/// Outputs: green below 70, yellow from 70 to 89, and red from 90.
fn pct_color(pct: u32) -> Color {
    match pct {
        90..=u32::MAX => Color::Red,
        70..=89 => Color::Yellow,
        _ => Color::Green,
    }
}

/// Purpose: This struct restores the terminal when the dashboard exits.
struct TermGuard;
impl Drop for TermGuard {
    /// Purpose: This function restores the terminal when the dashboard exits.
    ///
    /// Inputs: none.
    ///
    /// Outputs: none.
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(stdout(), LeaveAlternateScreen, Show);
    }
}

/// Purpose: This function runs the dashboard and prints any error before exiting with
/// status 1.
///
/// Inputs: none.
///
/// Outputs: none.
fn main() {
    if let Err(e) = run() {
        let _ = disable_raw_mode();
        let _ = execute!(stdout(), LeaveAlternateScreen, Show);
        eprintln!("jade-gpu-mon: {e}");
        std::process::exit(1);
    }
}

/// Purpose: This function polls the first NVIDIA GPU and redraws the dashboard every 500
/// ms until `q` or Ctrl-C.
///
/// Inputs: none.
///
/// Outputs: `Ok(())` on quit, or an NVML or terminal error.
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let nvml = Nvml::init()
        .map_err(|e| format!("NVML init failed: {e}\nIs an NVIDIA driver installed?"))?;
    let device = nvml.device_by_index(0)?;
    let gpu_name = device.name()?;

    enable_raw_mode()?;
    let _guard = TermGuard;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen, Hide)?;

    loop {
        let mem = device.memory_info()?;
        let util = device.utilization_rates()?;
        let mem_clk = device.clock_info(Clock::Memory).unwrap_or(0);
        let gpu_clk = device.clock_info(Clock::Graphics).unwrap_or(0);
        let temp = device.temperature(TemperatureSensor::Gpu).unwrap_or(0);
        let power_mw = device.power_usage().unwrap_or(0);
        let limit_mw = device
            .enforced_power_limit()
            .or_else(|_| device.power_management_limit())
            .unwrap_or(0);

        let vram_pct = ((mem.used as f64 / mem.total as f64) * 100.0) as u32;
        let bw_pct = util.memory;
        let gpu_pct = util.gpu;

        // fit bar to terminal width — leave room for label + percentage
        let (cols, _) = size().unwrap_or((80, 24));
        let bar_w = (cols as usize).saturating_sub(12).min(40).max(10);

        // ── render frame ──────────────────────────────────────────────────────
        // Queue clear + full frame together so they flush atomically (no flicker).
        queue!(out, MoveTo(0, 0), Clear(ClearType::All))?;

        // header
        queue!(
            out,
            SetAttribute(Attribute::Bold),
            SetForegroundColor(Color::Cyan),
            Print(format!(" JADE GPU MON  ·  {}\r\n\r\n", gpu_name)),
            SetAttribute(Attribute::Reset),
            ResetColor,
        )?;

        // VRAM
        queue!(
            out,
            SetForegroundColor(Color::White),
            Print(format!(
                " VRAM      {:>4.1} / {:>4.1} GB\r\n",
                mem.used as f64 / 1e9,
                mem.total as f64 / 1e9,
            )),
            Print(" "),
            ResetColor,
            SetForegroundColor(pct_color(vram_pct)),
            Print(bar(vram_pct, bar_w)),
            ResetColor,
            Print(format!("  {:>3}%\r\n\r\n", vram_pct)),
        )?;

        // Bandwidth
        queue!(
            out,
            SetForegroundColor(Color::White),
            Print(" Bandwidth\r\n"),
            Print(" "),
            ResetColor,
            SetForegroundColor(pct_color(bw_pct)),
            Print(bar(bw_pct, bar_w)),
            ResetColor,
            Print(format!("  {:>3}%\r\n\r\n", bw_pct)),
        )?;

        // GPU compute utilization
        queue!(
            out,
            SetForegroundColor(Color::White),
            Print(" GPU Util\r\n"),
            Print(" "),
            ResetColor,
            SetForegroundColor(pct_color(gpu_pct)),
            Print(bar(gpu_pct, bar_w)),
            ResetColor,
            Print(format!("  {:>3}%\r\n\r\n", gpu_pct)),
        )?;

        // Clocks / temp / power
        queue!(
            out,
            SetForegroundColor(Color::DarkGrey),
            Print(format!(
                " Mem Clock  {:>5} MHz    GPU Clock  {:>5} MHz\r\n",
                mem_clk, gpu_clk,
            )),
            Print(format!(
                " Temp       {:>3}°C       Power      {:.0}W / {:.0}W\r\n\r\n",
                temp,
                power_mw as f64 / 1000.0,
                limit_mw as f64 / 1000.0,
            )),
            Print(" q · quit    refreshes every 500ms\r\n"),
            ResetColor,
        )?;

        out.flush()?;

        // ── input ─────────────────────────────────────────────────────────────
        if poll(Duration::from_millis(POLL_MS))? {
            if let Event::Key(k) = read()? {
                match k.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('c')
                        if k.modifiers.contains(KeyModifiers::CONTROL) =>
                    {
                        break
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}
