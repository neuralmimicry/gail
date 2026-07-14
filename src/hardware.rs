use std::path::Path;

use serde::{Deserialize, Serialize};
use sysinfo::System;
use tokio::process::Command;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct GpuDevice {
    pub index: usize,
    pub name: String,
    /// Total framebuffer memory reported by `nvidia-smi` in MiB.
    pub memory_mb: u64,
    /// Free framebuffer memory reported by `nvidia-smi` in MiB.
    pub free_memory_mb: u64,
    /// Optional CUDA compute capability (for example `8.6`).
    pub compute_capability: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct HardwareProfile {
    pub cpu_cores: usize,
    pub cpu_arch: String,
    pub cpu_model: Option<String>,
    pub total_memory_mb: u64,
    pub available_memory_mb: u64,
    pub gpus: Vec<GpuDevice>,
}

impl HardwareProfile {
    pub fn gpu_count(&self) -> usize {
        self.gpus.len()
    }

    pub fn total_gpu_memory_mb(&self) -> u64 {
        self.gpus.iter().map(|gpu| gpu.memory_mb).sum()
    }

    pub fn total_gpu_free_memory_mb(&self) -> u64 {
        self.gpus.iter().map(|gpu| gpu.free_memory_mb).sum()
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
    system.refresh_cpu_all();
    let cpu_model = system
        .cpus()
        .first()
        .map(|cpu| cpu.brand().trim().to_string())
        .filter(|value| !value.is_empty());
    let total_memory_mb = system.total_memory().saturating_div(1024 * 1024);
    let available_memory_mb = system.available_memory().saturating_div(1024 * 1024);
    let gpus = detect_nvidia_gpus().await;
    HardwareProfile {
        cpu_cores,
        cpu_arch: std::env::consts::ARCH.to_string(),
        cpu_model,
        total_memory_mb,
        available_memory_mb,
        gpus,
    }
}

pub fn log_hardware_profile(component: &str, profile: &HardwareProfile) {
    tracing::info!(
        component,
        cpu_cores = profile.cpu_cores,
        cpu_arch = %profile.cpu_arch,
        cpu_model = profile.cpu_model.as_deref().unwrap_or("unknown"),
        total_memory_mb = profile.total_memory_mb,
        available_memory_mb = profile.available_memory_mb,
        gpu_count = profile.gpu_count(),
        gpu_memory_mb = profile.total_gpu_memory_mb(),
        gpu_free_memory_mb = profile.total_gpu_free_memory_mb(),
        "detected runtime hardware profile"
    );
    for gpu in &profile.gpus {
        tracing::info!(
            component,
            gpu_index = gpu.index,
            gpu_name = %gpu.name,
            gpu_memory_mb = gpu.memory_mb,
            gpu_free_memory_mb = gpu.free_memory_mb,
            gpu_compute_capability = gpu.compute_capability.as_deref().unwrap_or("unknown"),
            "detected GPU device"
        );
    }
}

async fn detect_nvidia_gpus() -> Vec<GpuDevice> {
    if !Path::new("/dev/nvidia0").exists() {
        return Vec::new();
    }

    if let Some(devices) = query_nvidia_gpus(
        "index,name,memory.total,memory.free,compute_cap",
        parse_nvidia_smi_line_with_free,
    )
    .await
    {
        return devices;
    }

    query_nvidia_gpus("name,memory.total", parse_nvidia_smi_line_legacy)
        .await
        .unwrap_or_default()
}

async fn query_nvidia_gpus(
    query_fields: &str,
    parser: fn(&str) -> Option<GpuDevice>,
) -> Option<Vec<GpuDevice>> {
    let query_arg = format!("--query-gpu={query_fields}");
    let output = Command::new("nvidia-smi")
        .args([query_arg.as_str(), "--format=csv,noheader,nounits"])
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(stdout.lines().filter_map(parser).collect::<Vec<_>>())
}

fn parse_nvidia_smi_line_with_free(line: &str) -> Option<GpuDevice> {
    let columns = line
        .split(',')
        .map(|value| value.trim())
        .collect::<Vec<_>>();
    if columns.len() < 5 {
        return None;
    }
    let index = columns[0].parse::<usize>().ok()?;
    let name = columns[1].trim();
    if name.is_empty() {
        return None;
    }
    let memory_mb = parse_positive_u64(columns[2]).unwrap_or(0);
    let free_memory_mb = parse_positive_u64(columns[3]).unwrap_or(0);
    let compute_capability = normalize_optional_string(columns[4]);
    Some(GpuDevice {
        index,
        name: name.to_string(),
        memory_mb,
        free_memory_mb,
        compute_capability,
    })
}

fn parse_nvidia_smi_line_legacy(line: &str) -> Option<GpuDevice> {
    let (name, memory) = line.split_once(',')?;
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    let memory_mb = parse_positive_u64(memory).unwrap_or(0);
    Some(GpuDevice {
        index: 0,
        name: name.to_string(),
        memory_mb,
        free_memory_mb: 0,
        compute_capability: None,
    })
}

fn parse_positive_u64(value: &str) -> Option<u64> {
    value.trim().parse::<u64>().ok().filter(|value| *value > 0)
}

fn normalize_optional_string(value: &str) -> Option<String> {
    let rendered = value.trim();
    if rendered.is_empty() {
        None
    } else {
        Some(rendered.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvidia_smi_csv_line_with_free_memory() {
        let parsed =
            parse_nvidia_smi_line_with_free("0, NVIDIA RTX A2000, 12288, 9216, 8.6").expect("gpu");
        assert_eq!(parsed.index, 0);
        assert_eq!(parsed.name, "NVIDIA RTX A2000");
        assert_eq!(parsed.memory_mb, 12_288);
        assert_eq!(parsed.free_memory_mb, 9_216);
        assert_eq!(parsed.compute_capability.as_deref(), Some("8.6"));
    }

    #[test]
    fn parses_legacy_nvidia_smi_csv_line() {
        let parsed = parse_nvidia_smi_line_legacy("NVIDIA RTX A2000, 12288").expect("gpu");
        assert_eq!(parsed.index, 0);
        assert_eq!(parsed.name, "NVIDIA RTX A2000");
        assert_eq!(parsed.memory_mb, 12_288);
        assert_eq!(parsed.free_memory_mb, 0);
        assert!(parsed.compute_capability.is_none());
    }
}
