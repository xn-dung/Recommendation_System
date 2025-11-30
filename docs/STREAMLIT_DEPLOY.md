# Deploying to Streamlit Cloud (with large external data)

If your processed datasets and/or model weights are too large for GitHub, the safest approach for Streamlit Cloud is to host them externally (S3/GCS/Azure or GitHub Releases) and point the app to those URLs.

1. Upload processed files to cloud storage (S3 example)

- Upload files `movies_processed.pkl` and `ratings_processed.pkl` to an S3 bucket
- Make them public or generate pre-signed URLs (recommended for private buckets)

2. Add secrets in Streamlit Cloud

- Go to your app on https://streamlit.io/cloud
- Open the "Secrets" section and paste keys like:

```
PROCESSED_MOVIES_URL = "https://bucket.s3.amazonaws.com/movies_processed.pkl"
PROCESSED_RATINGS_URL = "https://bucket.s3.amazonaws.com/ratings_processed.pkl"
```

3. The app will attempt to download missing files from these URLs at startup

- On the first run, the files will be streamed into `data/` — the repo will not store them in git.
- If you prefer to always keep models/data in cloud and avoid downloads, you can adapt the training or model loading to read directly from S3 with boto3.

4. Helpful tips

- Use pre-signed URLs when you want to keep data private and grant limited access.
- Avoid committing credentials to git. Use Streamlit secrets or environment variables in CI.
- If your repository contains large model files, Git LFS is another option, but costs and limits may apply on GitHub's free plans.

Local testing

- Locally you can set environment variables (Windows cmd):

```bat
set PROCESSED_MOVIES_URL=https://example.com/movies_processed.pkl
set PROCESSED_RATINGS_URL=https://example.com/ratings_processed.pkl
python -m streamlit run app.py
```
