use anyhow::Result;
use lancor::hub::HubClient;
use std::io::Write;

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

        _ => {
            println!("lancor — llama.cpp client + HuggingFace Hub model manager");
            println!();
            println!("Usage:");
            println!("  lancor pull <repo> [file]    Download a GGUF model from HF Hub");
            println!("  lancor list                  List cached models");
            println!("  lancor search <query>        Search HF Hub for models");
            println!("  lancor rm <repo> <file>      Delete a cached model");
        }
    }

    Ok(())
}
