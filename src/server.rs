//! llama.cpp binary orchestration — manage llama-server and CLI tools.
//!
//! Wraps llama-server, llama-cli, llama-quantize, llama-bench, and other
//! llama.cpp binaries with typed Rust APIs and builder-pattern flag construction.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use tokio::time::{sleep, Duration};

/// llama-server process manager.
///
/// Manages the lifecycle of a llama-server process, including starting,
/// stopping, and health checking. Provides an OpenAI-compatible API endpoint.
///
/// # Fields
///
/// * `process` - Internal child process handle (None if not running).
/// * `port` - Port number the server is listening on.
/// * `host` - Host address the server is bound to.
/// * `model_path` - Path to the GGUF model file being served.
#[derive(Debug)]
pub struct LlamaServer {
    process: Option<Child>,
    pub port: u16,
    pub host: String,
    pub model_path: PathBuf,
}

/// Builder for llama-server arguments.
///
/// Constructs command-line arguments for llama-server using a builder pattern.
/// All fields have sensible defaults; see individual builder methods for details.
///
/// # Example
///
/// ```
/// use lancor::server::ServerConfig;
/// let config = ServerConfig::new("model.gguf")
///     .host("0.0.0.0")
///     .port(8080)
///     .gpu_layers(40)
///     .ctx_size(4096);
/// ```
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
    /// Create a new ServerConfig with the specified model path.
    ///
    /// # Arguments
    ///
    /// * `model` - Path to the GGUF model file to load.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::ServerConfig;
    /// let config = ServerConfig::new("/path/to/model.gguf");
    /// ```
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Set the host address to bind the server to.
    ///
    /// Default: `127.0.0.1`
    ///
    /// # Arguments
    ///
    /// * `host` - Hostname or IP address to bind to.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::ServerConfig;
    /// let config = ServerConfig::new("model.gguf").host("0.0.0.0");
    /// ```
    pub fn host(mut self, host: impl Into<String>) -> Self { self.host = host.into(); self }

    /// Set the port number for the server to listen on.
    ///
    /// Default: `8080`
    ///
    /// # Arguments
    ///
    /// * `port` - Port number (0-65535).
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::ServerConfig;
    /// let config = ServerConfig::new("model.gguf").port(3000);
    /// ```
    pub fn port(mut self, port: u16) -> Self { self.port = port; self }

    /// Set the number of GPU layers to offload to the GPU.
    ///
    /// Default: `99` (offload all layers)
    ///
    /// # Arguments
    ///
    /// * `n` - Number of layers to offload. Use negative values for auto-detection.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::ServerConfig;
    /// let config = ServerConfig::new("model.gguf").gpu_layers(40);
    /// ```
    pub fn gpu_layers(mut self, n: i32) -> Self { self.n_gpu_layers = n; self }

    /// Set the context size (sequence length) for the model.
    ///
    /// Default: `8192`
    ///
    /// # Arguments
    ///
    /// * `n` - Context size in tokens.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::ServerConfig;
    /// let config = ServerConfig::new("model.gguf").ctx_size(4096);
    /// ```
    pub fn ctx_size(mut self, n: u32) -> Self { self.ctx_size = n; self }

    /// Set the number of parallel sequences to process.
    ///
    /// Default: `1`
    ///
    /// # Arguments
    ///
    /// * `n` - Number of parallel sequences.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::ServerConfig;
    /// let config = ServerConfig::new("model.gguf").parallel(4);
    /// ```
    pub fn parallel(mut self, n: u32) -> Self { self.n_parallel = n; self }

    /// Set the number of threads to use for computation.
    ///
    /// Default: `None` (auto-detect)
    ///
    /// # Arguments
    ///
    /// * `n` - Number of threads.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::ServerConfig;
    /// let config = ServerConfig::new("model.gguf").threads(8);
    /// ```
    pub fn threads(mut self, n: u32) -> Self { self.threads = Some(n); self }

    /// Set the batch size for prompt processing.
    ///
    /// Default: `None` (use default)
    ///
    /// # Arguments
    ///
    /// * `n` - Batch size.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::ServerConfig;
    /// let config = ServerConfig::new("model.gguf").batch_size(512);
    /// ```
    pub fn batch_size(mut self, n: u32) -> Self { self.batch_size = Some(n); self }

    /// Enable or disable Flash Attention optimization.
    ///
    /// Default: `false`
    ///
    /// # Arguments
    ///
    /// * `v` - `true` to enable Flash Attention.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::ServerConfig;
    /// let config = ServerConfig::new("model.gguf").flash_attn(true);
    /// ```
    pub fn flash_attn(mut self, v: bool) -> Self { self.flash_attn = v; self }

    /// Enable or disable memory locking (mlock) to prevent swapping.
    ///
    /// Default: `false`
    ///
    /// # Arguments
    ///
    /// * `v` - `true` to lock memory.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::ServerConfig;
    /// let config = ServerConfig::new("model.gguf").mlock(true);
    /// ```
    pub fn mlock(mut self, v: bool) -> Self { self.mlock = v; self }

    /// Set the API key for server authentication.
    ///
    /// Default: `None`
    ///
    /// # Arguments
    ///
    /// * `key` - API key string.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::ServerConfig;
    /// let config = ServerConfig::new("model.gguf").api_key("my-secret-key");
    /// ```
    pub fn api_key(mut self, key: impl Into<String>) -> Self { self.api_key = Some(key.into()); self }

    /// Add an extra command-line argument to pass to llama-server.
    ///
    /// # Arguments
    ///
    /// * `a` - Argument string (without leading dashes).
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::ServerConfig;
    /// let config = ServerConfig::new("model.gguf").arg("--some-flag").arg("value");
    /// ```
    pub fn arg(mut self, a: impl Into<String>) -> Self { self.extra_args.push(a.into()); self }

    /// Build the command-line arguments for llama-server.
    ///
    /// # Returns
    ///
    /// A vector of argument strings suitable for passing to `Command::args()`.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::ServerConfig;
    /// let config = ServerConfig::new("model.gguf").port(8080);
    /// let args = config.to_args();
    /// println!("Arguments: {:?}", args);
    /// ```
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

    /// Start llama-server with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Server configuration specifying model and runtime parameters.
    ///
    /// # Returns
    ///
    /// A `LlamaServer` instance managing the spawned process.
    ///
    /// # Errors
    ///
    /// Returns an error if the llama-server binary is not found on PATH or if
    /// the process fails to spawn.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lancor::server::{ServerConfig, LlamaServer};
    /// let config = ServerConfig::new("model.gguf").port(8080);
    /// let server = LlamaServer::start(&config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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

    /// Wait for the server to become healthy (ready to accept requests).
    ///
    /// Polls the `/health` endpoint until it responds successfully or the timeout expires.
    ///
    /// # Arguments
    ///
    /// * `timeout_secs` - Maximum number of seconds to wait.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the server became healthy within the timeout.
    ///
    /// # Errors
    ///
    /// Returns an error if the server does not become healthy within the specified timeout.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lancor::server::{ServerConfig, LlamaServer};
    /// let config = ServerConfig::new("model.gguf");
    /// let server = LlamaServer::start(&config)?;
    /// server.wait_healthy(30).await?;
    /// println!("Server is ready!");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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

    /// Get the OpenAI-compatible base URL for this server.
    ///
    /// # Returns
    ///
    /// A string in the format `http://host:port` suitable for use as an OpenAI API base URL.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lancor::server::{ServerConfig, LlamaServer};
    /// let config = ServerConfig::new("model.gguf").port(8080);
    /// let server = LlamaServer::start(&config)?;
    /// let base_url = server.base_url();
    /// println!("API endpoint: {}/v1/chat/completions", base_url);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    /// Stop the server.
    ///
    /// Attempts to gracefully terminate the llama-server process by sending a kill signal
    /// and waiting for it to exit. If the server is not running, this is a no-op.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the server was stopped or wasn't running.
    ///
    /// # Errors
    ///
    /// Returns an error if the kill signal fails or the process doesn't exit cleanly.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lancor::server::{ServerConfig, LlamaServer};
    /// let config = ServerConfig::new("model.gguf");
    /// let mut server = LlamaServer::start(&config)?;
    /// server.stop()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn stop(&mut self) -> Result<()> {
        if let Some(ref mut child) = self.process {
            child.kill().context("killing llama-server")?;
            child.wait().context("waiting for llama-server to exit")?;
        }
        self.process = None;
        Ok(())
    }

    /// Check if the server process is still running.
    ///
    /// This method polls the process status without blocking.
    ///
    /// # Returns
    ///
    /// `true` if the server process is alive, `false` if it has exited or was never started.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lancor::server::{ServerConfig, LlamaServer};
    /// let config = ServerConfig::new("model.gguf");
    /// let mut server = LlamaServer::start(&config)?;
    /// if server.is_running() {
    ///     println!("Server is alive!");
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
///
/// Constructs command-line arguments for llama-cli using a builder pattern.
///
/// # Example
///
/// ```
/// use lancor::server::CliConfig;
/// let config = CliConfig::new("model.gguf")
///     .prompt("Hello, world!")
///     .predict(50)
///     .temperature(0.8);
/// ```
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
    /// Create a new CliConfig with the specified model path.
    ///
    /// # Arguments
    ///
    /// * `model` - Path to the GGUF model file to use.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::CliConfig;
    /// let config = CliConfig::new("/path/to/model.gguf");
    /// ```
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

    /// Set the prompt text for generation.
    ///
    /// # Arguments
    ///
    /// * `p` - Prompt string.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::CliConfig;
    /// let config = CliConfig::new("model.gguf").prompt("Hello, world!");
    /// ```
    pub fn prompt(mut self, p: impl Into<String>) -> Self { self.prompt = Some(p.into()); self }

    /// Set the number of tokens to predict/generate.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of tokens to generate.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::CliConfig;
    /// let config = CliConfig::new("model.gguf").predict(128);
    /// ```
    pub fn predict(mut self, n: u32) -> Self { self.n_predict = Some(n); self }

    /// Set the temperature for sampling.
    ///
    /// Higher values increase randomness; lower values make output more deterministic.
    ///
    /// # Arguments
    ///
    /// * `t` - Temperature value (typically 0.0 - 2.0).
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::CliConfig;
    /// let config = CliConfig::new("model.gguf").temperature(0.7);
    /// ```
    pub fn temperature(mut self, t: f32) -> Self { self.temperature = Some(t); self }

    /// Enable interactive mode.
    ///
    /// In interactive mode, llama-cli will prompt for input repeatedly.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::CliConfig;
    /// let config = CliConfig::new("model.gguf").interactive();
    /// ```
    pub fn interactive(mut self) -> Self { self.interactive = true; self }

    /// Add an extra command-line argument to pass to llama-cli.
    ///
    /// # Arguments
    ///
    /// * `a` - Argument string (without leading dashes).
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::CliConfig;
    /// let config = CliConfig::new("model.gguf").arg("--some-flag").arg("value");
    /// ```
    pub fn arg(mut self, a: impl Into<String>) -> Self { self.extra_args.push(a.into()); self }

    /// Build the command-line arguments for llama-cli.
    ///
    /// # Returns
    ///
    /// A vector of argument strings suitable for passing to `Command::args()`.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::CliConfig;
    /// let config = CliConfig::new("model.gguf").prompt("test").predict(10);
    /// let args = config.to_args();
    /// ```
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
///
/// # Arguments
///
/// * `config` - Configuration specifying model and generation parameters.
///
/// # Returns
///
/// The standard output from llama-cli as a string.
///
/// # Errors
///
/// Returns an error if `llama-cli` binary is not found or if the command fails.
///
/// # Example
///
/// ```no_run
/// use lancor::server::{CliConfig, run_cli};
/// let config = CliConfig::new("model.gguf").prompt("Hello").predict(10);
/// let output = run_cli(&config)?;
/// println!("Output: {}", output);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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
///
/// Represents the various quantization formats that can be applied to GGUF models.
/// Each variant corresponds to a specific quantization method supported by llama.cpp.
///
/// See the [llama.cpp documentation](https://github.com/ggerganov/llama.cpp#quantization)
/// for details on each quantization type.
#[derive(Debug, Clone, Copy)]
pub enum QuantType {
    Q4_0, Q4_1, Q4_K_S, Q4_K_M,
    Q5_0, Q5_1, Q5_K_S, Q5_K_M,
    Q6_K, Q8_0,
    IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS,
    F16, F32,
}

impl QuantType {
    /// Get the string representation of this quantization type.
    ///
    /// Returns the command-line flag used by llama-quantize for this type.
    ///
    /// # Returns
    ///
    /// A static string slice like `"Q4_0"`, `"Q5_K_M"`, etc.
    ///
    /// # Example
    ///
    /// ```
    /// use lancor::server::QuantType;
    /// let qtype = QuantType::Q4_K_M;
    /// assert_eq!(qtype.as_str(), "Q4_K_M");
    /// ```
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
///
/// # Arguments
///
/// * `input` - Path to the input GGUF model file.
/// * `output` - Path where the quantized model will be written.
/// * `qtype` - Quantization type to apply.
///
/// # Returns
///
/// `Ok(())` if quantization succeeded.
///
/// # Errors
///
/// Returns an error if `llama-quantize` binary is not found or if the quantization fails.
///
/// # Example
///
/// ```no_run
/// use lancor::server::{quantize, QuantType};
/// quantize(
///     "model-f16.gguf",
///     "model-q4_k_m.gguf",
///     QuantType::Q4_K_M
/// )?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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
///
/// # Arguments
///
/// * `model` - Path to the GGUF model file to benchmark.
/// * `n_gpu_layers` - Number of GPU layers to offload.
/// * `ctx_size` - Context size (sequence length) for the benchmark.
///
/// # Returns
///
/// The standard output from llama-bench as a string.
///
/// # Errors
///
/// Returns an error if `llama-bench` binary is not found or if the benchmark fails.
///
/// # Example
///
/// ```no_run
/// use lancor::server::bench;
/// let output = bench("model.gguf", 40, 4096)?;
/// println!("Benchmark results:\n{}", output);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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
