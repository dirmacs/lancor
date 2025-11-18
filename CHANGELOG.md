# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of lancor
- Async Rust client for llama.cpp's OpenAI-compatible API
- Support for chat completions (streaming and non-streaming)
- Support for text completions
- Support for embeddings generation
- API key authentication
- Builder pattern for request construction
- Type-safe request/response handling
- Comprehensive examples and documentation

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - YYYY-MM-DD

### Added
- Initial public release
- `LlamaCppClient` for connecting to llama.cpp servers
- `ChatCompletionRequest` with builder pattern
- `CompletionRequest` for text completions
- `EmbeddingRequest` for generating embeddings
- Streaming support for chat completions
- API key authentication support
- Helper methods for creating system/user/assistant messages
- Comprehensive README with usage examples
- Basic usage example in `examples/` directory

[Unreleased]: https://github.com/yourusername/lancor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/lancor/releases/tag/v0.1.0