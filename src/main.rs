use anyhow::Result;
use lancor::bench;
use lancor::hub::HubClient;
use lancor::server::ServerConfig;
use std::io::Write;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("pull") => {
            let repo = args.get(2).map(|s| s.as_str())
                .unwrap_or("unsloth/Qwen3.5-35B-A3B-GGUF");
            let filename = args.get(3).map(|s| s.as_str());

            let hub = HubClient::new()?;

            // If no filename given, list available GGUFs
            let filename = match filename {
                Some(f) => f.to_string(),
                None => {
                    println!("Listing GGUF files in {}...", repo);
                    let files = hub.list_gguf(repo).await?;
                    if files.is_empty() {
                        eprintln!("No GGUF files found in {}", repo);
                        return Ok(());
                    }
                    for f in &files {
                        let size_mb = f.size.unwrap_or(0) as f64 / 1_048_576.0;
                        println!("  {:<50} {:>8.1} MB", f.filename, size_mb);
                    }
                    eprintln!("\nUsage: lancor pull {} <filename>", repo);
                    return Ok(());
                }
            };

            println!("Downloading {}/{}...", repo, filename);

            let progress: lancor::hub::ProgressFn = Box::new(|downloaded, total| {
                if total > 0 {
                    let pct = (downloaded as f64 / total as f64) * 100.0;
                    let dl_mb = downloaded as f64 / 1_048_576.0;
                    let tot_mb = total as f64 / 1_048_576.0;
                    eprint!("\r  {:.1}/{:.1} MB ({:.1}%)", dl_mb, tot_mb, pct);
                    let _ = std::io::stderr().flush();
                }
            });

            let path = hub.download(repo, &filename, Some(progress)).await?;
            eprintln!();
            println!("Saved: {}", path.display());
        }

        Some("list") => {
            let hub = HubClient::new()?;
            let cached = hub.list_cached()?;
            if cached.is_empty() {
                println!("No cached models. Use 'lancor pull <repo> <file>' to download.");
                return Ok(());
            }
            println!("Cached models:");
            for m in &cached {
                let size_gb = m.size as f64 / 1_073_741_824.0;
                println!("  {:<40} {:<40} {:.2} GB", m.repo_id, m.filename, size_gb);
                println!("    {}", m.path.display());
            }
        }

        Some("search") => {
            let query = args[2..].join(" ");
            let hub = HubClient::new()?;
            let results = hub.search(&query, 10).await?;
            for r in &results {
                println!("  {:<60} downloads={}", r.repo_id, r.downloads);
            }
        }

        Some("rm") => {
            let repo = args.get(2).expect("usage: lancor rm <repo> <file>");
            let file = args.get(3).expect("usage: lancor rm <repo> <file>");
            let hub = HubClient::new()?;
            hub.delete(repo, file).await?;
            println!("Deleted: {}/{}", repo, file);
        }

        Some("bench") => {
            // lancor bench <model_path> [--label NAME] [--port PORT] [--json]
            // lancor bench --all                  (bench all cached models)
            // lancor bench --url http://host:port  (bench against running server)
            let mut model_path: Option<PathBuf> = None;
            let mut label: Option<String> = None;
            let mut port: u16 = 8080;
            let mut json_output = false;
            let mut against_url: Option<String> = None;
            let mut bench_all = false;
            let mut ngl: i32 = 99;
            let mut ctx: u32 = 8192;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--label" => { i += 1; label = args.get(i).cloned(); }
                    "--port" => { i += 1; port = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(8080); }
                    "--json" => { json_output = true; }
                    "--url" => { i += 1; against_url = args.get(i).cloned(); }
                    "--all" => { bench_all = true; }
                    "--ngl" => { i += 1; ngl = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(99); }
                    "--ctx" => { i += 1; ctx = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(8192); }
                    arg if !arg.starts_with('-') && model_path.is_none() => {
                        model_path = Some(PathBuf::from(arg));
                    }
                    _ => {}
                }
                i += 1;
            }

            if let Some(ref url) = against_url {
                // Bench against a running server
                let lbl = label.unwrap_or_else(|| "running-server".into());
                let cfg = bench::BenchConfig::new(&lbl, model_path.unwrap_or_default())
                    .base_url(url);
                println!("Benchmarking against {}...", url);
                let result = bench::run_suite(&cfg).await?;
                if json_output {
                    println!("{}", bench::to_json(&[result])?);
                } else {
                    bench::print_table(&[result]);
                }
            } else if bench_all {
                // Bench all cached models
                let hub = HubClient::new()?;
                let cached = hub.list_cached()?;
                let gguf_models: Vec<_> = cached.into_iter()
                    .filter(|m| m.filename.ends_with(".gguf"))
                    .collect();

                if gguf_models.is_empty() {
                    println!("No cached GGUF models. Use 'lancor pull' first.");
                    return Ok(());
                }

                println!("Benchmarking {} cached models...\n", gguf_models.len());
                let models: Vec<_> = gguf_models.iter().map(|m| {
                    let lbl = m.filename.trim_end_matches(".gguf").to_string();
                    let scfg = ServerConfig::new(&m.path)
                        .port(port)
                        .gpu_layers(ngl)
                        .ctx_size(ctx);
                    (lbl, m.path.clone(), scfg)
                }).collect();

                let results = bench::compare(models).await?;
                if json_output {
                    println!("{}", bench::to_json(&results)?);
                } else {
                    bench::print_table(&results);
                }
            } else if let Some(path) = model_path {
                // Bench a single model
                let lbl = label.unwrap_or_else(|| {
                    path.file_stem().unwrap_or_default().to_string_lossy().to_string()
                });
                let scfg = ServerConfig::new(&path)
                    .port(port)
                    .gpu_layers(ngl)
                    .ctx_size(ctx);
                println!("Benchmarking {}...", lbl);
                let result = bench::run_suite_managed(&path, &lbl, scfg).await?;
                if json_output {
                    println!("{}", bench::to_json(&[result])?);
                } else {
                    bench::print_table(&[result]);
                }
            } else {
                println!("Usage:");
                println!("  lancor bench <model.gguf>        Bench a single model (manages server)");
                println!("  lancor bench --all               Bench all cached models");
                println!("  lancor bench --url http://h:p    Bench against a running server");
                println!("  lancor bench ... --json           Output JSON");
                println!("  lancor bench ... --ngl 99         GPU layers (default: 99)");
                println!("  lancor bench ... --ctx 8192       Context size (default: 8192)");
            }
        }

        _ => {
            println!("lancor — llama.cpp client + HuggingFace Hub + model benchmarking");
            println!();
            println!("Usage:");
            println!("  lancor pull <repo> [file]    Download a GGUF model from HF Hub");
            println!("  lancor list                  List cached models");
            println!("  lancor search <query>        Search HF Hub for models");
            println!("  lancor rm <repo> <file>      Delete a cached model");
            println!("  lancor bench <model|--all>   Benchmark models (5-test triage)");
        }
    }

    Ok(())
}
