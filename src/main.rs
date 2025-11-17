use anyhow::{Context, Result};
use futures::stream::StreamExt;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ============================================================================
// Request Types
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}
impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: String,
}

// ============================================================================
// Response Types
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoiceDelta>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatChoiceDelta {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionResponse {
    pub content: String,
    pub model: Option<String>,
    pub stop: Option<bool>,
    pub tokens_predicted: Option<u32>,
    pub tokens_evaluated: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: Option<u32>,
    pub total_tokens: u32,
}

// ============================================================================
// Client
// ============================================================================

#[derive(Debug, Clone)]
pub struct LlamaCppClient {
    http_client: HttpClient,
    base_url: String,
    api_key: Option<String>,
}

impl LlamaCppClient {
    /// Create a new client with the specified base URL
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let http_client = HttpClient::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            http_client,
            base_url: base_url.into(),
            api_key: None,
        })
    }

    /// Create a new client with the specified base URL and API key
    pub fn with_api_key(base_url: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        let http_client = HttpClient::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            http_client,
            base_url: base_url.into(),
            api_key: Some(api_key.into()),
        })
    }

    /// Create a client connecting to localhost:8080
    pub fn default() -> Result<Self> {
        Self::new("http://localhost:8080")
    }

    /// Send a chat completion request
    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut req = self.http_client.post(&url).json(&request);

        if let Some(api_key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = req
            .send()
            .await
            .context("Failed to send chat completion request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API error ({}): {}", status, error_text);
        }

        response
            .json()
            .await
            .context("Failed to parse chat completion response")
    }

    /// Send a streaming chat completion request
    pub async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<impl futures::Stream<Item = Result<ChatCompletionChunk>>> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut req = self.http_client.post(&url).json(&request);

        if let Some(api_key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = req
            .send()
            .await
            .context("Failed to send streaming chat completion request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API error({}): {}", status, error_text);
        }

        let stream = response.bytes_stream().map(|result| {
            let bytes = result.context("Failed to read stream chunk")?;
            let text = String::from_utf8_lossy(&bytes);

            for line in text.lines() {
                if line.starts_with("data: ") {
                    let data = &line[6..];
                    if data == "[DONE]" {
                        continue;
                    }
                    let chunk: ChatCompletionChunk =
                        serde_json::from_str(data).context("Failed to parse chunk")?;
                    return Ok(chunk);
                }
            }

            anyhow::bail!("No valid data in chunk")
        });

        Ok(stream)
    }

    /// Send a text completion request
    pub async fn completion(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let url = format!("{}/v1/completions", self.base_url);

        let mut req = self.http_client.post(&url).json(&request);

        if let Some(api_key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = req
            .send()
            .await
            .context("Failed to send completion request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API error ({}): {}", status, error_text);
        }

        response
            .json()
            .await
            .context("Failed to parse completion response")
    }

    /// Send an embedding request
    pub async fn embedding(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let url = format!("{}/v1/embeddings", self.base_url);

        let mut req = self.http_client.post(&url).json(&request);

        if let Some(api_key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = req
            .send()
            .await
            .context("Failed to send embedding request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API error ({}): {}", status, error_text);
        }

        response
            .json()
            .await
            .context("Failed to parse embedding response")
    }
}

// ============================================================================
// Builder Pattern for Requests
// ============================================================================
//
//

impl ChatCompletionRequest {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: Vec::new(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            stream: None,
            stop: None,
        }
    }

    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }
}

impl CompletionRequest {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            temperature: None,
            max_tokens: None,
            stream: None,
        }
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }
}

impl EmbeddingRequest {
    pub fn new(model: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
        }
    }
}

// ============================================================================
// Usage Examples
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize client
    let client = LlamaCppClient::with_api_key("http://localhost:1337", "jafong")?;

    // Example 1: Simple chat completion
    println!("=== Chat Completion Example ===");
    let request = ChatCompletionRequest::new("Qwen3-VL-2B-Instruct-IQ4_XS")
        .message(Message::system("You are a helpful assistant."))
        .message(Message::user("What is Rust programming language?"))
        .max_tokens(100)
        .temperature(0.7);

    let response = client.chat_completion(request).await?;
    println!("Response: {}", response.choices[0].message.content);
    println!("Tokens used: {}", response.usage.total_tokens);

    // Example 2: Streaming chat completion
    println!("\n=== Streaming Chat Completion Example ===");
    let streaming_request = ChatCompletionRequest::new("Qwen3-VL-2B-Instruct-IQ4_XS")
        .message(Message::user("Count from 1 to 5."))
        .stream(true)
        .max_tokens(50);

    let mut stream = client.chat_completion_stream(streaming_request).await?;
    print!("Streaming response: ");
    while let Some(chunk_result) = stream.next().await {
        if let Ok(chunk) = chunk_result {
            if let Some(content) = &chunk.choices[0].delta.content {
                print!("{}", content);
            }
        }
    }
    println!();

    // Example 3: Text completion
    println!("\n=== Text Completion Example ===");
    let completion_request =
        CompletionRequest::new("Qwen3-VL-2B-Instruct-IQ4_XS", "The quick brown fox")
            .max_tokens(20)
            .temperature(0.8);

    let completion_response = client.completion(completion_request).await?;
    println!("Completion: {}", completion_response.content);

    // Example 4: Embeddings
    println!("\n=== Embedding Example ===");
    let embedding_request = EmbeddingRequest::new("Qwen3-VL-2B-Instruct-IQ4_XS", "Hello, world!");

    let embedding_response = client.embedding(embedding_request).await?;
    println!(
        "Embedding dimension: {:?}",
        embedding_response.data[0].embedding.len()
    );
    println!(
        "First 5 values: {:?}",
        &embedding_response.data[0].embedding[..5]
    );

    Ok(())
}
