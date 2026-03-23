<p align="center">
  <img src="docs/static/img/lancor-logo.svg" width="128" alt="lancor">
</p>

<h1 align="center">Lancor</h1>

<p align="center">
  End-to-end llama.cpp toolkit in Rust.<br>
  API client, HuggingFace Hub, server orchestration, 5-test benchmark suite.
</p>

<p align="center">
  <a href="https://crates.io/crates/lancor"><img src="https://img.shields.io/crates/v/lancor.svg" alt="crates.io"></a>
  <a href="https://docs.rs/lancor"><img src="https://docs.rs/lancor/badge.svg" alt="docs.rs"></a>
  <img src="https://img.shields.io/badge/license-GPL--3.0-blue.svg" alt="GPL-3.0">
</p>

---

## Features

### Library
- **LlamaCppClient**: Async OpenAI-compatible API client (chat/completion/embeddings)
- **HubClient**: Pure Rust HuggingFace Hub downloads with progress callbacks
- **Server orchestration**: Programmatic llama-server lifecycle management
- **Benchmark suite**: 5-test triage (throughput, tool calls, codegen, reasoning)

### CLI
```bash
lancor pull <repo> [file]     # Download GGUF from HF Hub
lancor list                   # List cached models
lancor search <query>         # Search HF Hub
lancor rm <repo> <file>       # Delete cached model
lancor bench <model|--all>    # Run benchmark suite
```

## Installation

```toml
[dependencies]
lancor = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
```

## Quick Start

```rust
use lancor::{LlamaCppClient, ChatCompletionRequest, Message};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = LlamaCppClient::new("http://localhost:8080")?;

    let request = ChatCompletionRequest::new("model-name")
        .message(Message::system("You are a helpful assistant."))
        .message(Message::user("What is Rust?"))
        .max_tokens(100);

    let response = client.chat_completion(request).await?;
    println!("{}", response.choices[0].message.content);
    Ok(())
}
```

## LlamaCppClient

OpenAI-compatible client for llama.cpp server (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`).

```rust
use lancor::{LlamaCppClient, ChatCompletionRequest, CompletionRequest, EmbeddingRequest, Message};

// Create client
let client = LlamaCppClient::new("http://localhost:8080")?;
let client = LlamaCppClient::with_api_key("http://localhost:8080", "sk-...")?;
let client = LlamaCppClient::default()?;  // localhost:8080

// Chat completion (non-streaming)
let request = ChatCompletionRequest::new("model")
    .message(Message::user("Explain quantum computing"))
    .temperature(0.7)
    .max_tokens(200);
let response = client.chat_completion(request).await?;

// Streaming chat completion
let request = ChatCompletionRequest::new("model")
    .message(Message::user("Write a short poem"))
    .stream(true)
    .max_tokens(100);
let mut stream = client.chat_completion_stream(request).await?;
while let Some(chunk) = stream.next().await {
    if let Some(content) = &chunk.choices[0].delta.content {
        print!("{}", content);
    }
}

// Text completion
let request = CompletionRequest::new("model", "Once upon a time")
    .max_tokens(50)
    .temperature(0.8);
let response = client.completion(request).await?;

// Embeddings
let request = EmbeddingRequest::new("model", "Hello, world!");
let response = client.embedding(request).await?;
let embedding = &response.data[0].embedding;
```

**Request builders support:** `temperature`, `max_tokens`, `top_p`, `stream`, `stop`, `chat_template_kwargs` (for chat) and `prompt` (for completion).

## HuggingFace Hub

Download and manage GGUF models directly from HuggingFace Hub.

```rust
use lancor::hub::{HubClient, ProgressFn};

// Create client (auto-detects HF_TOKEN or ~/.cache/huggingface/token)
let hub = HubClient::new()?;

// Search models
let results = hub.search("qwen3.5 gguf", 10).await?;
for r in results {
    println!("{} (downloads: {})", r.repo_id, r.downloads);
}

// List GGUF files in a repo
let files = hub.list_gguf("unsloth/Qwen3.5-35B-A3B-GGUF").await?;
for f in files {
    let size_mb = f.size.unwrap_or(0) as f64 / 1_048_576.0;
    println!("{} ({:.1} MB)", f.filename, size_mb);
}

// Download with progress
let progress: ProgressFn = Box::new(|downloaded, total| {
    let pct = (downloaded as f64 / total as f64) * 100.0;
    eprint!("\r{:.1}%", pct);
});
let path = hub.download("unsloth/Qwen3.5-35B-A3B-GGUF", "model-Q4_K_M.gguf", Some(progress)).await?;
println!("Saved: {}", path.display());

// List cached models
let cached = hub.list_cached()?;
for m in cached {
    println!("{}: {} ({:.2} GB)", m.repo_id, m.filename, m.size as f64 / 1_073_741_824.0);
}

// Delete cached model
hub.delete("unsloth/Qwen3.5-35B-A3B-GGUF", "model-Q4_K_M.gguf").await?;
```

**Cache directory**: `~/.cache/lancor/models/` (configurable via `HubClient::with_cache_dir(path)`).

## Server Orchestration

Programmatic control over llama-server, llama-cli, llama-quantize, and llama-bench.

### LlamaServer

```rust
use lancor::server::{LlamaServer, ServerConfig};

// Configure server
let config = ServerConfig::new("model-Q4_K_M.gguf")
    .host("127.0.0.1")
    .port(8080)
    .gpu_layers(99)      // Offload layers to GPU
    .ctx_size(8192)      // Context length
    .parallel(1)         // Parallel sequences
    .threads(4)          // CPU threads
    .batch_size(512)     // Batch size for prompt processing
    .flash_attn(true)    // Enable flash attention
    .mlock(true)         // Lock model in RAM
    .api_key("sk-...")   // Require API key
    .arg("--some-flag"); // Extra args

// Start server
let mut server = LlamaServer::start(&config)?;
server.wait_healthy(60).await?;
println!("Server ready at: {}", server.base_url());

// Use with client
let client = lancor::LlamaCppClient::new(server.base_url())?;
// ... make requests

// Stop server
server.stop()?;
```

**ServerConfig defaults**: `host=127.0.0.1`, `port=8080`, `n_gpu_layers=99`, `ctx_size=8192`, `n_parallel=1`, `cont_batching=true`, `metrics=true`.

### LlamaCli

Run inference with llama-cli (captures stdout):

```rust
use lancor::server::CliConfig;

let config = CliConfig::new("model-Q4_K_M.gguf")
    .prompt("What is Rust?")
    .predict(100)
    .temperature(0.7)
    .interactive();  // Enable interactive mode

let output = lancor::server::run_cli(&config)?;
println!("{}", output);
```

### Quantization

```rust
use lancor::server::{quantize, QuantType};

quantize(
    "model-f32.gguf",
    "model-Q4_K_M.gguf",
    QuantType::Q4_K_M,
)?;
```

**Supported QuantType**: `Q4_0`, `Q4_1`, `Q4_K_S`, `Q4_K_M`, `Q5_0`, `Q5_1`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0`, `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, `F16`, `F32`.

### Raw llama-bench wrapper

```rust
use lancor::server::bench;

let output = bench("model.gguf", 99, 8192)?;
println!("{}", output);
```

## Benchmark Suite

5-test triage for comparing model quantizations and sizes.

```rust
use lancor::bench::{run_suite_managed, BenchConfig, print_table};
use lancor::server::ServerConfig;

// Single model (auto-starts/stops server)
let result = run_suite_managed(
    &std::path::Path::new("model-Q4_K_M.gguf"),
    "Qwen3.5-35B-Q4_K_M",
    ServerConfig::new("model-Q4_K_M.gguf")
        .gpu_layers(99)
        .ctx_size(8192),
).await?;

// Against existing server
let cfg = BenchConfig::new("my-model", "model.gguf")
    .base_url("http://localhost:8080");
let result = lancor::bench::run_suite(&cfg).await?;

// Compare multiple models
let models = vec![
    ("Q4_K_M", path1, ServerConfig::new(&path1).gpu_layers(99)),
    ("Q8_0", path2, ServerConfig::new(&path2).gpu_layers(99)),
];
let results = lancor::bench::compare(models).await?;
print_table(&results);
```

**Benchmark tests:**
- **Throughput**: tokens/s for prompt processing and generation
- **Tool call**: single function call accuracy
- **Multi-tool**: parallel tool invocation (min 5 tools)
- **Codegen**: fizzbuzz implementation (score 0-4)
- **Reasoning**: logic puzzle correctness

**Output example:**
```
┌──────────────────┬───────┬──────────┬──────────┬──────┬───────┬──────┬───────────┐
│ Model            │ Size  │ PP tok/s │ TG tok/s │ Tool │ Multi │ Code │ Reasoning │
├──────────────────┼───────┼──────────┼──────────┼──────┼───────┼──────┼───────────┤
│ Qwen3.5-35B-Q4_K │  20.1G│     45.2 │    128.7 │  ✓   │ 5/5   │ 4/4  │     ✓     │
└──────────────────┴───────┴──────────┴──────────┴──────┴───────┴──────┴───────────┘
```

**JSON export**: `lancor::bench::to_json(results)`.

## CLI Reference

### `lancor pull <repo> [file]`

Download a GGUF model from HuggingFace Hub.

```bash
# List available GGUF files in a repo
lancor pull unsloth/Qwen3.5-35B-A3B-GGUF

# Download specific file
lancor pull unsloth/Qwen3.5-35B-A3B-GGUF model-Q4_K_M.gguf
```

### `lancor list`

List all cached models.

```bash
lancor list
# Output:
# unsloth/Qwen3.5-35B-A3B-GGUF: model-Q4_K_M.gguf (20.12 GB)
#   /home/user/.cache/lancor/models/unsloth--Qwen3.5-35B-A3B-GGUF/model-Q4_K_M.gguf
```

### `lancor search <query>`

Search HuggingFace Hub for models.

```bash
lancor search "qwen3.5 gguf"
# Output:
# unsloth/Qwen3.5-35B-A3B-GGUF                     downloads=12345
# ...
```

### `lancor rm <repo> <file>`

Delete a cached model file.

```bash
lancor rm unsloth/Qwen3.5-35B-A3B-GGUF model-Q4_K_M.gguf
```

### `lancor bench <model|--all> [options]`

Run the benchmark suite.

```bash
# Benchmark a single model (auto-manages server)
lancor bench model-Q4_K_M.gguf --label "MyModel-Q4" --ngl 99 --ctx 8192

# Benchmark all cached models
lancor bench --all --ngl 99 --port 8081

# Benchmark against existing server
lancor bench --url http://localhost:8080 --label "Remote" model.gguf

# JSON output
lancor bench model.gguf --json > results.json
```

**Benchmark options:**
- `--label NAME` — Model label for results table
- `--port PORT` — Server port (default: 8080, for auto-managed)
- `--ngl LAYERS` — GPU layers (default: 99)
- `--ctx SIZE` — Context size (default: 8192)
- `--url URL` — Use existing server instead of starting one
- `--all` — Benchmark all cached GGUF models
- `--json` — Output JSON instead of table

## Requirements

- Rust 1.91+
- llama.cpp binaries on PATH: `llama-server`, `llama-cli`, `llama-quantize`, `llama-bench`
- For HubClient: network access to huggingface.co

## Running llama-server manually

```bash
./server -m model.gguf --port 8080 --api-key sk-... --metrics --cont-batching
```

Then use `LlamaCppClient` to interact with it.

## Ecosystem

| Project | What |
|---------|------|
| [ares](https://github.com/dirmacs/ares) | Agentic AI server — uses lancor for local llama.cpp inference |
| [pawan](https://github.com/dirmacs/pawan) | Self-healing CLI coding agent |
| [daedra](https://dirmacs.github.io/daedra) | Web search MCP server |
| [thulp](https://dirmacs.github.io/thulp) | Execution context engineering |

Built by [DIRMACS](https://dirmacs.com).

## License

GPL-3.0
