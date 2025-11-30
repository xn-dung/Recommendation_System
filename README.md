# Cinema AI Pro — Movie Recommendation System

This repo contains a Streamlit application and training pipeline for a MovieLens-based recommendation system.

Key points for deployment when you have large datasets:

- Do NOT commit the raw or processed dataset files if they are too large for GitHub.
- Use one of the methods documented in `docs/DATA_DEPLOYMENT.md` — Git LFS, GitHub Releases, or cloud storage with pre-signed URLs.
- The app supports automatic download of processed data when you configure environment variables or Streamlit secrets: `PROCESSED_MOVIES_URL`, `PROCESSED_RATINGS_URL` (or `PROCESSED_DATA_URL` for a shared URL).

See `docs/DATA_DEPLOYMENT.md` for examples and step-by-step guidance.
