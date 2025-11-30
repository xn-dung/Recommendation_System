# Handling large datasets & models for deployment

When datasets or model files are too large to commit to GitHub, do NOT add them to the repo. Instead choose one of the options below — the repository includes a small automation to download processed files at runtime if you provide URLs via environment variables or Streamlit secrets.

Recommended options (short):

- Git LFS — convenient for moderately large files (but subject to storage & bandwidth limits on GitHub/Git LFS). Good for model binaries < a few GB and if you want files under the same repository.
- GitHub Releases — upload large files as release assets and use a release download URL or pre-signed URL from CI. Releases can be a simpler alternative to LFS.
- Cloud storage (best for large datasets): upload processed files to S3 / GCP / Azure Blob and use a pre-signed URL or a short-lived authenticated URL. This scales and avoids GitHub limits.
- Public hosting (e.g., Google Drive, Dropbox) — acceptable for public datasets; use gdown or public HTTPS links when needed.

How the app in this repo helps:

- The Streamlit app attempts to auto-download `data/movies_processed.pkl` and `data/ratings_processed.pkl` when they are missing and you set download URLs.
- Provide download URLs either as environment variables (PROCESSED_MOVIES_URL, PROCESSED_RATINGS_URL or PROCESSED_DATA_URL) or use Streamlit secrets when deploying to Streamlit Cloud.

Example (Streamlit Cloud secrets):

1. In Streamlit Cloud -> App -> Secrets, add keys:

```
PROCESSED_MOVIES_URL = "https://example-bucket.s3.amazonaws.com/movies_processed.pkl"
PROCESSED_RATINGS_URL = "https://example-bucket.s3.amazonaws.com/ratings_processed.pkl"
```

2. Deploy the app — on startup, the app will attempt to download and cache these files into `data/` automatically.

If you prefer not to rely on runtime downloads, create a small sample dataset for local development and document how to generate or download the full dataset via `train_pipeline.py` (the pipeline can recreate processed files locally if raw data is available).

Security note:

- Keep credentials (AWS keys, GCS keys) out of the repo. Use Streamlit secrets or environment variables for credentials and pre-signed URLs when possible.

Troubleshooting:

- If automatic download fails, the app will show an error with advice to run `python train_pipeline.py` locally or provide the correct download URLs.
