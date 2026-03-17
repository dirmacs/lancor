//! HuggingFace Hub client — download GGUF models directly from HF.
//!
//! Pure Rust, no Python. Uses the HF API + reqwest for streaming downloads
//! with progress callbacks. Caches models in `~/.cache/lancor/models/`.

use anyhow::{Context, Result, bail};
use futures::stream::StreamExt;
use reqwest::Client as HttpClient;
use serde::Deserialize;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

/// Default cache directory for downloaded models.
pub fn default_cache_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("lancor")
        .join("models")
}

/// File metadata from HuggingFace API.
#[derive(Debug, Clone, Deserialize)]
pub struct HubFile {
    #[serde(rename = "rfilename")]
    pub filename: String,
    #[serde(default)]
    pub size: Option<u64>,
}

/// Model info from HuggingFace API.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    #[serde(rename = "id")]
    pub repo_id: String,
    #[serde(default)]
    pub siblings: Vec<HubFile>,
}

/// Search result from HuggingFace API.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchResult {
    #[serde(rename = "id")]
    pub repo_id: String,
    #[serde(default)]
    pub downloads: u64,
}

/// Progress callback type.
pub type ProgressFn = Box<dyn Fn(u64, u64) + Send + Sync>;

/// HuggingFace Hub client for model downloads.
pub struct HubClient {
    http: HttpClient,
    cache_dir: PathBuf,
    token: Option<String>,
}

impl HubClient {
    /// Create a new HubClient with default cache directory.
    pub fn new() -> Result<Self> {
        Self::with_cache_dir(default_cache_dir())
    }

    /// Create a new HubClient with a custom cache directory.
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self> {
        let http = HttpClient::builder()
            .user_agent("lancor/0.1 (Rust)")
            .timeout(std::time::Duration::from_secs(3600))
            .build()
            .context("building HTTP client")?;

        // Check for HF token in env or ~/.cache/huggingface/token
        let token = std::env::var("HF_TOKEN").ok().or_else(|| {
            let token_path = dirs::home_dir()?.join(".cache/huggingface/token");
            std::fs::read_to_string(token_path).ok().map(|t| t.trim().to_string())
        });

        Ok(Self { http, cache_dir, token })
    }

    fn auth_headers(&self) -> Vec<(&str, String)> {
        match &self.token {
            Some(t) => vec![("Authorization", format!("Bearer {}", t))],
            None => vec![],
        }
    }

    /// Search for models on HuggingFace Hub.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let url = format!(
            "https://huggingface.co/api/models?search={}&sort=downloads&direction=-1&limit={}",
            urlencoding(query), limit
        );
        let mut req = self.http.get(&url);
        for (k, v) in self.auth_headers() {
            req = req.header(k, v);
        }
        let resp = req.send().await.context("HF search request")?;
        if !resp.status().is_success() {
            bail!("HF search failed: {}", resp.status());
        }
        resp.json().await.context("parsing search results")
    }

    /// Get model info including file listing.
    pub async fn model_info(&self, repo_id: &str) -> Result<ModelInfo> {
        let url = format!("https://huggingface.co/api/models/{}", repo_id);
        let mut req = self.http.get(&url);
        for (k, v) in self.auth_headers() {
            req = req.header(k, v);
        }
        let resp = req.send().await.context("HF model info request")?;
        if !resp.status().is_success() {
            bail!("model '{}' not found: {}", repo_id, resp.status());
        }
        resp.json().await.context("parsing model info")
    }

    /// List GGUF files in a repository.
    pub async fn list_gguf(&self, repo_id: &str) -> Result<Vec<HubFile>> {
        let info = self.model_info(repo_id).await?;
        Ok(info.siblings.into_iter()
            .filter(|f| f.filename.ends_with(".gguf"))
            .collect())
    }

    /// Get the local cache path for a model file.
    pub fn cache_path(&self, repo_id: &str, filename: &str) -> PathBuf {
        let safe_repo = repo_id.replace('/', "--");
        self.cache_dir.join(&safe_repo).join(filename)
    }

    /// Check if a model file is already cached.
    pub fn is_cached(&self, repo_id: &str, filename: &str) -> bool {
        self.cache_path(repo_id, filename).exists()
    }

    /// Download a file from HuggingFace Hub with progress reporting.
    ///
    /// Returns the local path to the downloaded file.
    /// Skips download if the file is already cached.
    pub async fn download(
        &self,
        repo_id: &str,
        filename: &str,
        progress: Option<ProgressFn>,
    ) -> Result<PathBuf> {
        let dest = self.cache_path(repo_id, filename);

        // Check cache
        if dest.exists() {
            let meta = std::fs::metadata(&dest)?;
            if meta.len() > 0 {
                if let Some(ref cb) = progress {
                    cb(meta.len(), meta.len());
                }
                return Ok(dest);
            }
        }

        // Create parent dirs
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await
                .with_context(|| format!("creating {}", parent.display()))?;
        }

        // Build download URL
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, filename
        );

        let mut req = self.http.get(&url);
        for (k, v) in self.auth_headers() {
            req = req.header(k, v);
        }

        let resp = req.send().await
            .with_context(|| format!("downloading {}", url))?;

        if !resp.status().is_success() {
            bail!("download failed: {} ({})", resp.status(), url);
        }

        let total = resp.content_length().unwrap_or(0);
        let mut downloaded: u64 = 0;

        // Stream to temp file, then rename (atomic-ish)
        let tmp_path = dest.with_extension("part");
        let mut file = tokio::fs::File::create(&tmp_path).await
            .with_context(|| format!("creating {}", tmp_path.display()))?;

        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("reading download stream")?;
            file.write_all(&chunk).await.context("writing to file")?;
            downloaded += chunk.len() as u64;
            if let Some(ref cb) = progress {
                cb(downloaded, total);
            }
        }

        file.flush().await?;
        drop(file);

        // Rename .part → final
        tokio::fs::rename(&tmp_path, &dest).await
            .with_context(|| format!("renaming {} → {}", tmp_path.display(), dest.display()))?;

        Ok(dest)
    }

    /// Delete a cached model file.
    pub async fn delete(&self, repo_id: &str, filename: &str) -> Result<()> {
        let path = self.cache_path(repo_id, filename);
        if path.exists() {
            tokio::fs::remove_file(&path).await
                .with_context(|| format!("deleting {}", path.display()))?;
        }
        Ok(())
    }

    /// List all cached models.
    pub fn list_cached(&self) -> Result<Vec<CachedModel>> {
        let mut models = Vec::new();
        if !self.cache_dir.exists() {
            return Ok(models);
        }
        for entry in std::fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() { continue; }
            let repo_name = entry.file_name().to_string_lossy().replace("--", "/");
            for file in std::fs::read_dir(entry.path())? {
                let file = file?;
                let fname = file.file_name().to_string_lossy().to_string();
                if fname.ends_with(".gguf") {
                    let size = file.metadata()?.len();
                    models.push(CachedModel {
                        repo_id: repo_name.clone(),
                        filename: fname,
                        path: file.path(),
                        size,
                    });
                }
            }
        }
        Ok(models)
    }
}

/// A cached model on disk.
#[derive(Debug, Clone)]
pub struct CachedModel {
    pub repo_id: String,
    pub filename: String,
    pub path: PathBuf,
    pub size: u64,
}

/// Simple URL encoding for query params.
fn urlencoding(s: &str) -> String {
    s.replace(' ', "+")
        .replace('/', "%2F")
        .replace(':', "%3A")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_path_format() {
        let client = HubClient::with_cache_dir(PathBuf::from("/tmp/test-cache")).unwrap();
        let path = client.cache_path("unsloth/Qwen3.5-35B-A3B-GGUF", "model-Q4_K_M.gguf");
        assert_eq!(
            path,
            PathBuf::from("/tmp/test-cache/unsloth--Qwen3.5-35B-A3B-GGUF/model-Q4_K_M.gguf")
        );
    }

    #[tokio::test]
    async fn search_works() {
        let client = HubClient::new().unwrap();
        let results = client.search("qwen3.5 gguf", 3).await.unwrap();
        assert!(!results.is_empty());
    }
}
