# lancor — Agent Guidelines

## What This Is

Lancor is a Rust toolkit for llama.cpp — model download, server management, API client, and benchmarking. It provides an OpenAI-compatible interface to local models.

## For Agents

- Run `cargo test` before changes
- HuggingFace Hub client uses progress callbacks — don't break the download UX
- Benchmark suite tests 5 scenarios — keep all passing
- API client follows OpenAI spec — don't deviate from the standard
- Model cache is at `~/.cache/lancor/models/` — never hardcode paths
