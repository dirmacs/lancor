//! Model benchmarking suite for llama.cpp GGUF models.
//!
//! Runs a standardized 5-test triage against a running llama-server:
//! throughput, single tool call, multi-tool, code generation, reasoning.
//! Designed for comparing quantizations and model sizes on Apple Silicon.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::server::{LlamaServer, ServerConfig};
use crate::{ChatCompletionRequest, LlamaCppClient, Message};

/// Result of a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub model_label: String,
    pub model_path: String,
    pub size_bytes: u64,
    pub quantization: String,
    pub throughput: ThroughputResult,
    pub tool_call: ToolCallResult,
    pub multi_tool: MultiToolResult,
    pub codegen: CodegenResult,
    pub reasoning: ReasoningResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputResult {
    pub prompt_tok_s: f64,
    pub gen_tok_s: f64,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub passed: bool,
    pub tool_name: String,
    pub tool_args: String,
    pub expected_name: String,
    pub expected_path_contains: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiToolResult {
    pub tool_count: usize,
    pub tools_called: Vec<String>,
    pub expected_min: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodegenResult {
    pub score: u8,
    pub has_fn_main: bool,
    pub has_loop: bool,
    pub has_modulo: bool,
    pub has_fizzbuzz: bool,
    pub code_preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningResult {
    pub passed: bool,
    pub answer: String,
    pub expected: String,
}

/// Configuration for a bench run.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub base_url: String,
    pub model_label: String,
    pub model_path: PathBuf,
    pub max_tokens: u32,
    /// Disable thinking mode for Qwen3.5 models.
    pub disable_thinking: bool,
}

impl BenchConfig {
    pub fn new(label: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self {
            base_url: "http://127.0.0.1:8080".to_string(),
            model_label: label.into(),
            model_path: path.into(),
            max_tokens: 512,
            disable_thinking: true,
        }
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

/// Run the full 5-test benchmark suite against a running server.
pub async fn run_suite(cfg: &BenchConfig) -> Result<BenchResult> {
    let client = LlamaCppClient::new(&cfg.base_url)?;
    let size = std::fs::metadata(&cfg.model_path)
        .map(|m| m.len())
        .unwrap_or(0);

    let quant = extract_quant(&cfg.model_label);

    let tp = test_throughput(&client, cfg).await?;
    let tc = test_tool_call(&client, cfg).await?;
    let mt = test_multi_tool(&client, cfg).await?;
    let cg = test_codegen(&client, cfg).await?;
    let rs = test_reasoning(&client, cfg).await?;

    Ok(BenchResult {
        model_label: cfg.model_label.clone(),
        model_path: cfg.model_path.to_string_lossy().to_string(),
        size_bytes: size,
        quantization: quant,
        throughput: tp,
        tool_call: tc,
        multi_tool: mt,
        codegen: cg,
        reasoning: rs,
    })
}

/// Run the full suite with automatic server lifecycle management.
/// Starts llama-server, waits for health, runs tests, stops server.
pub async fn run_suite_managed(
    model_path: &Path,
    label: &str,
    server_cfg: ServerConfig,
) -> Result<BenchResult> {
    let mut server = LlamaServer::start(&server_cfg)?;
    server.wait_healthy(60).await?;

    let cfg = BenchConfig::new(label, model_path)
        .base_url(server.base_url());

    let result = run_suite(&cfg).await;

    server.stop()?;
    result
}

/// Compare multiple models — starts/stops server for each.
pub async fn compare(models: Vec<(String, PathBuf, ServerConfig)>) -> Result<Vec<BenchResult>> {
    let mut results = Vec::new();
    for (label, path, server_cfg) in models {
        match run_suite_managed(&path, &label, server_cfg).await {
            Ok(r) => results.push(r),
            Err(e) => {
                eprintln!("  ✗ {} failed: {}", label, e);
            }
        }
    }
    Ok(results)
}

/// Print results as a formatted table.
pub fn print_table(results: &[BenchResult]) {
    println!("┌──────────────────┬───────┬──────────┬──────────┬──────┬───────┬──────┬───────────┐");
    println!("│ Model            │ Size  │ PP tok/s │ TG tok/s │ Tool │ Multi │ Code │ Reasoning │");
    println!("├──────────────────┼───────┼──────────┼──────────┼──────┼───────┼──────┼───────────┤");
    for r in results {
        let size_gb = r.size_bytes as f64 / 1_073_741_824.0;
        let tool = if r.tool_call.passed { "✓" } else { "✗" };
        let multi = format!("{}/{}", r.multi_tool.tool_count, r.multi_tool.expected_min);
        let code = format!("{}/4", r.codegen.score);
        let reason = if r.reasoning.passed { "✓" } else { "✗" };
        println!(
            "│ {:<16} │ {:>4.1}G │ {:>8.1} │ {:>8.1} │  {}   │ {:>5} │ {:>4} │     {}     │",
            r.model_label, size_gb,
            r.throughput.prompt_tok_s, r.throughput.gen_tok_s,
            tool, multi, code, reason
        );
    }
    println!("└──────────────────┴───────┴──────────┴──────────┴──────┴───────┴──────┴───────────┘");
}

/// Export results as JSON.
pub fn to_json(results: &[BenchResult]) -> Result<String> {
    serde_json::to_string_pretty(results).context("serializing bench results")
}

// ─── Individual Tests ───────────────────────────────────────────────────────

async fn test_throughput(client: &LlamaCppClient, cfg: &BenchConfig) -> Result<ThroughputResult> {
    let req = build_request(
        cfg,
        None,
        "Explain how a CPU cache hierarchy works: L1, L2, L3 caches, sizes, and cache coherency.",
        cfg.max_tokens,
    );

    let t0 = Instant::now();
    let resp = client.chat_completion(req).await?;
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let comp = resp.usage.completion_tokens.unwrap_or(0);

    Ok(ThroughputResult {
        prompt_tok_s: resp.usage.prompt_tokens as f64 / (total_ms / 1000.0),
        gen_tok_s: comp as f64 / (total_ms / 1000.0),
        prompt_tokens: resp.usage.prompt_tokens,
        completion_tokens: comp,
        total_ms,
    })
}

async fn test_tool_call(client: &LlamaCppClient, cfg: &BenchConfig) -> Result<ToolCallResult> {
    let payload = serde_json::json!({
        "model": "test",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
            {"role": "user", "content": "Read the file at path src/main.rs"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "File path"}},
                    "required": ["path"]
                }
            }
        }],
        "tool_choice": "auto",
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": false}
    });

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", cfg.base_url))
        .json(&payload)
        .send()
        .await?;

    let body: serde_json::Value = resp.json().await?;
    let tc = &body["choices"][0]["message"]["tool_calls"];

    if let Some(calls) = tc.as_array() {
        if let Some(first) = calls.first() {
            let name = first["function"]["name"].as_str().unwrap_or("").to_string();
            let args = first["function"]["arguments"].as_str().unwrap_or("{}").to_string();
            let args_parsed: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
            let path = args_parsed["path"].as_str().unwrap_or("").to_string();
            let passed = name == "read_file" && path.contains("main.rs");
            return Ok(ToolCallResult {
                passed,
                tool_name: name,
                tool_args: args,
                expected_name: "read_file".into(),
                expected_path_contains: "main.rs".into(),
            });
        }
    }

    Ok(ToolCallResult {
        passed: false,
        tool_name: "none".into(),
        tool_args: "{}".into(),
        expected_name: "read_file".into(),
        expected_path_contains: "main.rs".into(),
    })
}

async fn test_multi_tool(client: &LlamaCppClient, cfg: &BenchConfig) -> Result<MultiToolResult> {
    let payload = serde_json::json!({
        "model": "test",
        "messages": [
            {"role": "system", "content": "You are a coding assistant. Complete the task using the provided tools."},
            {"role": "user", "content": "First read config.toml, then write output.txt with content hello world"}
        ],
        "tools": [
            {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Write content to file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}
        ],
        "tool_choice": "auto",
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": false}
    });

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", cfg.base_url))
        .json(&payload)
        .send()
        .await?;

    let body: serde_json::Value = resp.json().await?;
    let tc = &body["choices"][0]["message"]["tool_calls"];

    let mut tools = Vec::new();
    if let Some(calls) = tc.as_array() {
        for call in calls {
            if let Some(name) = call["function"]["name"].as_str() {
                tools.push(name.to_string());
            }
        }
    }

    Ok(MultiToolResult {
        tool_count: tools.len(),
        tools_called: tools,
        expected_min: 2,
    })
}

async fn test_codegen(client: &LlamaCppClient, cfg: &BenchConfig) -> Result<CodegenResult> {
    let req = build_request(
        cfg,
        Some("Rust programmer. Output ONLY code. No markdown, no explanation."),
        "Write a complete Rust main function for FizzBuzz 1-15. Print FizzBuzz for multiples of 15, Fizz for 3, Buzz for 5, otherwise the number.",
        512,
    );

    let resp = client.chat_completion(req).await?;
    let code = &resp.choices[0].message.content;

    let has_fn = code.contains("fn main");
    let has_loop = code.contains("for ") || code.contains("while ");
    let has_mod = code.contains("% 3") || code.contains("%3") || code.contains("% 5") || code.contains("% 15");
    let has_fb = code.to_lowercase().contains("fizzbuzz");
    let score = has_fn as u8 + has_loop as u8 + has_mod as u8 + has_fb as u8;

    let preview: String = code.lines().take(8).collect::<Vec<_>>().join("\n");

    Ok(CodegenResult {
        score,
        has_fn_main: has_fn,
        has_loop,
        has_modulo: has_mod,
        has_fizzbuzz: has_fb,
        code_preview: preview,
    })
}

async fn test_reasoning(client: &LlamaCppClient, cfg: &BenchConfig) -> Result<ReasoningResult> {
    let req = build_request(
        cfg,
        None,
        "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Answer with just the number.",
        64,
    );

    let resp = client.chat_completion(req).await?;
    let answer = resp.choices[0].message.content.trim().to_string();
    let passed = answer.contains('9') && !answer.contains("19") && !answer.contains("90");

    Ok(ReasoningResult {
        passed,
        answer,
        expected: "9".into(),
    })
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn build_request(
    cfg: &BenchConfig,
    system: Option<&str>,
    user: &str,
    max_tokens: u32,
) -> ChatCompletionRequest {
    let mut req = ChatCompletionRequest::new("test");
    if let Some(sys) = system {
        req = req.message(Message::system(sys));
    }
    req = req.message(Message::user(user)).max_tokens(max_tokens);
    if cfg.disable_thinking {
        req = req.disable_thinking();
    }
    req
}

fn extract_quant(label: &str) -> String {
    let parts: Vec<&str> = label.split('-').collect();
    for p in parts.iter().rev() {
        let up = p.to_uppercase();
        if up.starts_with('Q') || up.starts_with("IQ") || up == "BF16" || up == "F16" {
            return up;
        }
    }
    "unknown".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_quant_from_label() {
        assert_eq!(extract_quant("4B-Q4_K_M"), "Q4_K_M");
        assert_eq!(extract_quant("9B-Q8_0"), "Q8_0");
        assert_eq!(extract_quant("9B-IQ4_XS"), "IQ4_XS");
    }
}
