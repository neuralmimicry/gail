use std::path::Path;

use serde::{Deserialize, Serialize};
use sysinfo::System;
use tokio::process::Command;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct GpuDevice {
    pub name: String,
    pub memory_mb: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct HardwareProfile {
    pub cpu_cores: usize,
    pub total_memory_mb: u64,
    pub gpus: Vec<GpuDevice>,
}

impl HardwareProfile {
    pub fn gpu_count(&self) -> usize {
        self.gpus.len()
    }

    pub fn total_gpu_memory_mb(&self) -> u64 {
        self.gpus.iter().map(|gpu| gpu.memory_mb).sum()
    }

    pub fn preferred_worker_threads(&self) -> usize {
        if self.cpu_cores <= 2 {
            1
        } else {
            (self.cpu_cores - 1).clamp(1, 256)
        }
    }
}

pub async fn detect_hardware() -> HardwareProfile {
    let cpu_cores = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let mut system = System::new();
    system.refresh_memory();
    let total_memory_mb = system.total_memory().saturating_div(1024 * 1024);
    let gpus = detect_nvidia_gpus().await;
    HardwareProfile {
        cpu_cores,
        total_memory_mb,
        gpus,
    }
}

pub fn log_hardware_profile(component: &str, profile: &HardwareProfile) {
    tracing::info!(
        component,
        cpu_cores = profile.cpu_cores,
        total_memory_mb = profile.total_memory_mb,
        gpu_count = profile.gpu_count(),
        gpu_memory_mb = profile.total_gpu_memory_mb(),
        "detected runtime hardware profile"
    );
    for gpu in &profile.gpus {
        tracing::info!(
            component,
            gpu_name = %gpu.name,
            gpu_memory_mb = gpu.memory_mb,
            "detected GPU device"
        );
    }
}

async fn detect_nvidia_gpus() -> Vec<GpuDevice> {
    if !Path::new("/dev/nvidia0").exists() {
        return Vec::new();
    }
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await;
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .filter_map(parse_nvidia_smi_line)
        .collect::<Vec<_>>()
}

fn parse_nvidia_smi_line(line: &str) -> Option<GpuDevice> {
    let (name, memory) = line.split_once(',')?;
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    let memory_mb = memory
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or(0);
    Some(GpuDevice {
        name: name.to_string(),
        memory_mb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvidia_smi_csv_line() {
        let parsed = parse_nvidia_smi_line("NVIDIA RTX A2000, 12288").expect("gpu");
        assert_eq!(parsed.name, "NVIDIA RTX A2000");
        assert_eq!(parsed.memory_mb, 12_288);
    }
}
