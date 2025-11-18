# lancor

A Rust client library for [llama.cpp](https://github.com/ggerganov/llama.cpp)'s OpenAI-compatible API server.

[![Crates.io](https://img.shields.io/crates/v/lancor.svg)](https://crates.io/crates/lancor)
[![Documentation](https://docs.rs/lancor/badge.svg)](https://docs.rs/lancor)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](LICENSE)

## Features

- 🚀 Async/await support with Tokio
- 💬 Chat completions (streaming and non-streaming)
- 📝 Text completions
- 🔢 Embeddings generation
- 🔑 API key authentication support
- 🎯 Type-safe request/response handling
- 🛠️ Builder pattern for easy request construction

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
lancor = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
```

## Quick Start

```rust
use lancor::{LlamaCppClient, ChatCompletionRequest, Message};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Create a client
    let client = LlamaCppClient::new("http://localhost:8080")?;
    
    // Build a chat completion request
    let request = ChatCompletionRequest::new("your-model-name")
        .message(Message::system("You are a helpful assistant."))
        .message(Message::user("What is Rust?"))
        .max_tokens(100);
    
    // Send the request
    let response = client.chat_completion(request).await?;
    println!("{}", response.choices[0].message.content);
    
    Ok(())
}
```

## Usage Examples

### Chat Completion

```rust
use lancor::{LlamaCppClient, ChatCompletionRequest, Message};

let client = LlamaCppClient::new("http://localhost:8080")?;

let request = ChatCompletionRequest::new("model-name")
    .message(Message::system("You are a helpful assistant."))
    .message(Message::user("Explain quantum computing"))
    .temperature(0.7)
    .max_tokens(200);

let response = client.chat_completion(request).await?;
println!("{}", response.choices[0].message.content);
```

### Streaming Chat Completion

```rust
use lancor::{LlamaCppClient, ChatCompletionRequest, Message};
use futures::stream::StreamExt;

let client = LlamaCppClient::new("http://localhost:8080")?;

let request = ChatCompletionRequest::new("model-name")
    .message(Message::user("Write a short poem"))
    .stream(true)
    .max_tokens(100);

let mut stream = client.chat_completion_stream(request).await?;

while let Some(chunk_result) = stream.next().await {
    if let Ok(chunk) = chunk_result {
        if let Some(content) = &chunk.choices[0].delta.content {
            print!("{}", content);
        }
    }
}
```

### Text Completion

```rust
use lancor::{LlamaCppClient, CompletionRequest};

let client = LlamaCppClient::new("http://localhost:8080")?;

let request = CompletionRequest::new("model-name", "Once upon a time")
    .max_tokens(50)
    .temperature(0.8);

let response = client.completion(request).await?;
println!("{}", response.content);
```

### Embeddings

```rust
use lancor::{LlamaCppClient, EmbeddingRequest};

let client = LlamaCppClient::new("http://localhost:8080")?;

let request = EmbeddingRequest::new("model-name", "Hello, world!");

let response = client.embedding(request).await?;
let embedding_vector = &response.data[0].embedding;
println!("Embedding dimension: {}", embedding_vector.len());
```

### Authentication

```rust
use lancor::LlamaCppClient;

// With API key
let client = LlamaCppClient::with_api_key(
    "http://localhost:8080",
    "your-api-key"
)?;
```

## API Reference

### `LlamaCppClient`

The main client for interacting with llama.cpp server.

#### Methods

- `new(base_url)` - Create a new client
- `with_api_key(base_url, api_key)` - Create a client with API key authentication
- `default()` - Create a client connecting to `http://localhost:8080`
- `chat_completion(request)` - Send a chat completion request
- `chat_completion_stream(request)` - Send a streaming chat completion request
- `completion(request)` - Send a text completion request
- `embedding(request)` - Send an embedding request

### Request Builders

All request types support a fluent builder pattern:

```rust
ChatCompletionRequest::new("model")
    .message(Message::user("Hello"))
    .temperature(0.7)
    .max_tokens(100)
    .top_p(0.9)
    .stream(true);
```

## Requirements

- Rust 1.70 or later
- A running llama.cpp server with OpenAI-compatible API enabled

## Running llama.cpp Server

To use this client, you need to run llama.cpp with the `--api-key` flag (optional) and ensure the OpenAI-compatible endpoints are enabled:

```bash
./server -m your-model.gguf --port 8080
```

## Examples

Check out the [examples](examples/) directory for more usage examples:

```bash
cargo run --example basic_usage
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - The amazing llama.cpp project
- [OpenAI API](https://platform.openai.com/docs/api-reference) - API specification reference