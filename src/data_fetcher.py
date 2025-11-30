"""
Small helper utilities to download dataset/model files at runtime if they are not present.

This supports simple public HTTP(S) or pre-signed URLs. For Google Drive / S3 with auth,
you should provide pre-signed URLs or add additional auth code (boto3 / gdown etc.).
"""
from __future__ import annotations

import os
import shutil
import requests


def download_file(url: str, dest_path: str, chunk_size: int = 1 << 20) -> None:
    """Download a file from a URL (streamed) to dest_path.

    Raises requests.HTTPError for non-200 responses. Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Use streaming download to avoid loading the whole file into memory
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        # Write to temporary file then move into place to avoid partial files
        tmp_path = dest_path + ".partial"
        with open(tmp_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        # All good — move into place
        shutil.move(tmp_path, dest_path)


def ensure_file_from_env(dest_path: str, env_vars: tuple[str, ...]) -> bool:
    """Ensure dest_path exists by downloading from first available URL found in env_vars.

    env_vars: a tuple of environment variable names to check (first found is used).
    Returns True if dest_path exists or was successfully downloaded, False otherwise.
    """
    if os.path.exists(dest_path):
        return True

    for ev in env_vars:
        url = os.environ.get(ev)
        if url:
            try:
                download_file(url, dest_path)
                return True
            except Exception:
                # If download fails, continue to next env var
                continue

    return False
