# lancor

End-to-end llama.cpp toolkit. API client, HuggingFace Hub integration, server orchestration, benchmarks.

## Build & Test

```bash
cargo build --release
cargo test
```

## CLI Commands

```bash
lancor pull <repo>          # download model from HuggingFace
lancor list                 # list cached models
lancor search <query>       # search HuggingFace Hub
lancor rm <repo>            # remove cached model
lancor bench <model>        # run benchmark suite
```

## Conventions

- Git author: `bkataru <baalateja.k@gmail.com>`
- OpenAI-compatible API (chat/completion/embeddings)
- Cache: `~/.cache/lancor/models/`
- Requires llama.cpp binaries on PATH
- Benchmark suite: throughput, tool calls, codegen, reasoning
