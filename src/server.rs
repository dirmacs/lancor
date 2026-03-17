//! llama.cpp binary orchestration — manage llama-server and CLI tools.
//!
//! Wraps llama-server, llama-cli, llama-quantize, llama-bench, and other
//! llama.cpp binaries with typed Rust APIs and builder-pattern flag construction.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use tokio::time::{sleep, Duration};

/// llama-server process manager.
pub struct LlamaServer {
    process: Option<Child>,
    pub port: u16,
    pub host: String,
    pub model_path: PathBuf,
}

/// Builder for llama-server arguments.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub model: PathBuf,
    pub host: String,
    pub port: u16,
    pub n_gpu_layers: i32,
    pub ctx_size: u32,
    pub n_parallel: u32,
    pub threads: Option<u32>,
    pub batch_size: Option<u32>,
    pub flash_attn: bool,
    pub mlock: bool,
    pub cont_batching: bool,
    pub metrics: bool,
    pub api_key: Option<String>,
    pub extra_args: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            host: "127.0.0.1".to_string(),
            port: 8080,
            n_gpu_layers: 99,
            ctx_size: 8192,
            n_parallel: 1,
            threads: None,
            batch_size: None,
            flash_attn: false,
            mlock: false,
            cont_batching: true,
            metrics: true,
            api_key: None,
            extra_args: vec![],
        }
    }
}

impl ServerConfig {
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    pub fn host(mut self, host: impl Into<String>) -> Self { self.host = host.into(); self }
    pub fn port(mut self, port: u16) -> Self { self.port = port; self }
    pub fn gpu_layers(mut self, n: i32) -> Self { self.n_gpu_layers = n; self }
    pub fn ctx_size(mut self, n: u32) -> Self { self.ctx_size = n; self }
    pub fn parallel(mut self, n: u32) -> Self { self.n_parallel = n; self }
    pub fn threads(mut self, n: u32) -> Self { self.threads = Some(n); self }
    pub fn batch_size(mut self, n: u32) -> Self { self.batch_size = Some(n); self }
    pub fn flash_attn(mut self, v: bool) -> Self { self.flash_attn = v; self }
    pub fn mlock(mut self, v: bool) -> Self { self.mlock = v; self }
    pub fn api_key(mut self, key: impl Into<String>) -> Self { self.api_key = Some(key.into()); self }
    pub fn arg(mut self, a: impl Into<String>) -> Self { self.extra_args.push(a.into()); self }

    /// Build the command-line arguments for llama-server.
    pub fn to_args(&self) -> Vec<String> {
        let mut args = vec![
            "-m".into(), self.model.to_string_lossy().to_string(),
            "--host".into(), self.host.clone(),
            "--port".into(), self.port.to_string(),
            "-ngl".into(), self.n_gpu_layers.to_string(),
            "-c".into(), self.ctx_size.to_string(),
            "-np".into(), self.n_parallel.to_string(),
        ];
        if let Some(t) = self.threads { args.extend(["-t".into(), t.to_string()]); }
        if let Some(b) = self.batch_size { args.extend(["-b".into(), b.to_string()]); }
        if self.flash_attn { args.push("-fa".into()); }
        if self.mlock { args.push("--mlock".into()); }
        if self.cont_batching { args.push("--cont-batching".into()); }
        if self.metrics { args.push("--metrics".into()); }
        if let Some(ref key) = self.api_key { args.extend(["--api-key".into(), key.clone()]); }
        args.extend(self.extra_args.clone());
        args
    }
}

impl LlamaServer {
    /// Find the llama-server binary on PATH.
    fn find_binary() -> Result<PathBuf> {
        which::which("llama-server").context("llama-server not found on PATH — install llama.cpp")
    }

    /// Start llama-server with the given config.
    pub fn start(config: &ServerConfig) -> Result<Self> {
        let bin = Self::find_binary()?;
        let args = config.to_args();

        let child = Command::new(&bin)
            .args(&args)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawning llama-server: {} {}", bin.display(), args.join(" ")))?;

        Ok(Self {
            process: Some(child),
            port: config.port,
            host: config.host.clone(),
            model_path: config.model.clone(),
        })
    }

    /// Wait for the server to be healthy (up to timeout_secs).
    pub async fn wait_healthy(&self, timeout_secs: u64) -> Result<()> {
        let url = format!("http://{}:{}/health", self.host, self.port);
        let client = reqwest::Client::new();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);

        loop {
            if tokio::time::Instant::now() > deadline {
                bail!("llama-server did not become healthy within {}s", timeout_secs);
            }
            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                _ => sleep(Duration::from_millis(500)).await,
            }
        }
    }

    /// Get the OpenAI-compatible base URL.
    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    /// Stop the server.
    pub fn stop(&mut self) -> Result<()> {
        if let Some(ref mut child) = self.process {
            child.kill().context("killing llama-server")?;
            child.wait().context("waiting for llama-server to exit")?;
        }
        self.process = None;
        Ok(())
    }

    /// Check if still running.
    pub fn is_running(&mut self) -> bool {
        match &mut self.process {
            Some(child) => child.try_wait().ok().flatten().is_none(),
            None => false,
        }
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

// ============================================================================
// llama-cli wrapper
// ============================================================================

/// Builder for llama-cli (interactive/completion mode).
#[derive(Debug, Clone)]
pub struct CliConfig {
    pub model: PathBuf,
    pub prompt: Option<String>,
    pub n_gpu_layers: i32,
    pub ctx_size: u32,
    pub n_predict: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub threads: Option<u32>,
    pub interactive: bool,
    pub extra_args: Vec<String>,
}

impl CliConfig {
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            prompt: None,
            n_gpu_layers: 99,
            ctx_size: 4096,
            n_predict: None,
            temperature: None,
            top_p: None,
            repeat_penalty: None,
            threads: None,
            interactive: false,
            extra_args: vec![],
        }
    }

    pub fn prompt(mut self, p: impl Into<String>) -> Self { self.prompt = Some(p.into()); self }
    pub fn predict(mut self, n: u32) -> Self { self.n_predict = Some(n); self }
    pub fn temperature(mut self, t: f32) -> Self { self.temperature = Some(t); self }
    pub fn interactive(mut self) -> Self { self.interactive = true; self }
    pub fn arg(mut self, a: impl Into<String>) -> Self { self.extra_args.push(a.into()); self }

    pub fn to_args(&self) -> Vec<String> {
        let mut args = vec![
            "-m".into(), self.model.to_string_lossy().to_string(),
            "-ngl".into(), self.n_gpu_layers.to_string(),
            "-c".into(), self.ctx_size.to_string(),
        ];
        if let Some(ref p) = self.prompt { args.extend(["-p".into(), p.clone()]); }
        if let Some(n) = self.n_predict { args.extend(["-n".into(), n.to_string()]); }
        if let Some(t) = self.temperature { args.extend(["--temp".into(), t.to_string()]); }
        if let Some(t) = self.top_p { args.extend(["--top-p".into(), t.to_string()]); }
        if let Some(r) = self.repeat_penalty { args.extend(["--repeat-penalty".into(), r.to_string()]); }
        if let Some(t) = self.threads { args.extend(["-t".into(), t.to_string()]); }
        if self.interactive { args.push("-i".into()); }
        args.extend(self.extra_args.clone());
        args
    }
}

/// Run llama-cli and capture output.
pub fn run_cli(config: &CliConfig) -> Result<String> {
    let bin = which::which("llama-cli").context("llama-cli not found")?;
    let output = Command::new(&bin)
        .args(config.to_args())
        .output()
        .context("running llama-cli")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("llama-cli failed: {}", stderr);
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

// ============================================================================
// llama-quantize wrapper
// ============================================================================

/// Quantization types supported by llama-quantize.
#[derive(Debug, Clone, Copy)]
pub enum QuantType {
    Q4_0, Q4_1, Q4_K_S, Q4_K_M,
    Q5_0, Q5_1, Q5_K_S, Q5_K_M,
    Q6_K, Q8_0,
    IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS,
    F16, F32,
}

impl QuantType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0", Self::Q4_1 => "Q4_1",
            Self::Q4_K_S => "Q4_K_S", Self::Q4_K_M => "Q4_K_M",
            Self::Q5_0 => "Q5_0", Self::Q5_1 => "Q5_1",
            Self::Q5_K_S => "Q5_K_S", Self::Q5_K_M => "Q5_K_M",
            Self::Q6_K => "Q6_K", Self::Q8_0 => "Q8_0",
            Self::IQ2_XXS => "IQ2_XXS", Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS", Self::IQ3_S => "IQ3_S",
            Self::IQ4_NL => "IQ4_NL", Self::IQ4_XS => "IQ4_XS",
            Self::F16 => "F16", Self::F32 => "F32",
        }
    }
}

/// Quantize a GGUF model.
pub fn quantize(input: &Path, output: &Path, qtype: QuantType) -> Result<()> {
    let bin = which::which("llama-quantize").context("llama-quantize not found")?;
    let status = Command::new(&bin)
        .args([
            input.to_string_lossy().as_ref(),
            output.to_string_lossy().as_ref(),
            qtype.as_str(),
        ])
        .status()
        .context("running llama-quantize")?;

    if !status.success() {
        bail!("llama-quantize failed with exit code {:?}", status.code());
    }
    Ok(())
}

// ============================================================================
// llama-bench wrapper
// ============================================================================

/// Run llama-bench and return raw output.
pub fn bench(model: &Path, n_gpu_layers: i32, ctx_size: u32) -> Result<String> {
    let bin = which::which("llama-bench").context("llama-bench not found")?;
    let output = Command::new(&bin)
        .args([
            "-m", &model.to_string_lossy(),
            "-ngl", &n_gpu_layers.to_string(),
            "-c", &ctx_size.to_string(),
        ])
        .output()
        .context("running llama-bench")?;

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}
