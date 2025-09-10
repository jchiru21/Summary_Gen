import os
import re
from pathlib import Path
from typing import List, Optional
from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "Chiranjeevijoshi/tripletgen-reranker-summary"

def list_checkpoints_on_hf(token: Optional[str] = None) -> List[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=HF_REPO, token=token)
    return [f for f in files if re.match(r"reranker_ckpt_.*\.pt$", f)]

def ensure_checkpoint(filename: str, out_dir: str = "checkpoints_clean", token: Optional[str] = None) -> str:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    local_path = out_dir_p / filename
    if local_path.exists():
        return str(local_path)
    print(f"[hf_utils] Downloading {filename} from {HF_REPO} ...")
    hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=str(out_dir_p), token=token)
    if not local_path.exists():
        raise RuntimeError(f"Failed to download {filename} from HF repo {HF_REPO}")
    return str(local_path)

def ensure_latest_checkpoint(out_dir: str = "checkpoints_clean", token: Optional[str] = None) -> str:
    """Find a checkpoint on HF (newest by lexicographic timestamp) and download it if local missing."""
    cands = list_checkpoints_on_hf(token=token)
    if not cands:
        raise RuntimeError("No checkpoint files found in HF repo: " + HF_REPO)
    # assume naming includes timestamp; sort descending and pick first
    chosen = sorted(cands, reverse=True)[0]
    return ensure_checkpoint(chosen, out_dir=out_dir, token=token)
