use anyhow::Result;
use futures::stream::StreamExt;
use lancor::{ChatCompletionRequest, CompletionRequest, EmbeddingRequest, LlamaCppClient, Message};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize client
    let client = LlamaCppClient::new("http://localhost:8080")?;

    // Or with API key:
    // let client = LlamaCppClient::with_api_key("http://localhost:8080", "your-api-key")?;

    // Example 1: Simple chat completion
    println!("=== Chat Completion Example ===");
    let request = ChatCompletionRequest::new("your-model-name")
        .message(Message::system("You are a helpful assistant."))
        .message(Message::user("What is Rust programming language?"))
        .max_tokens(100)
        .temperature(0.7);

    let response = client.chat_completion(request).await?;
    println!("Response: {}", response.choices[0].message.content);
    println!("Tokens used: {}", response.usage.total_tokens);

    // Example 2: Streaming chat completion
    println!("\n=== Streaming Chat Completion Example ===");
    let streaming_request = ChatCompletionRequest::new("your-model-name")
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
    let completion_request = CompletionRequest::new("your-model-name", "The quick brown fox")
        .max_tokens(20)
        .temperature(0.8);

    let completion_response = client.completion(completion_request).await?;
    println!("Completion: {}", completion_response.content);

    // Example 4: Embeddings
    println!("\n=== Embedding Example ===");
    let embedding_request = EmbeddingRequest::new("your-model-name", "Hello, world!");

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
