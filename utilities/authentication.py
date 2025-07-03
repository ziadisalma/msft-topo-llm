import os
import subprocess
from pathlib import Path

def find_repo_root(start = None):
    start = (start or Path(__file__)).resolve()
    for p in [start, *start.parents]:
        if (p / ".git").is_dir():
            return p
    # Fallback to git CLI
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start.parent,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return Path(root)
    except:
        raise RuntimeError("Cannot locate the repository root.")

def get_hf_token():
    repo_root = find_repo_root()
    token_file = os.path.join(repo_root.parent, ".huggingface", "HF_TOKEN")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            hf_token = Path(token_file).expanduser().read_text().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"No HF_TOKEN in environment and token file not found at {token_file!r}")
    return hf_token
