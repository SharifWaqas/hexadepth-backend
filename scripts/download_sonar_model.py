"""Install HexaDepth's private DRISHTI YOLO11n checkpoint on this machine.

Requires huggingface_hub and read access to ShaarifW/hexadepth-checkpoints.
Never commit the .pt file or share Hugging Face credentials.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_REMOTE = "hexadepth-v2/run/weights/best.pt"
BUCKET = "ShaarifW/hexadepth-checkpoints"
DEST = Path(__file__).resolve().parents[1] / "models" / "hexadepth_sonar_best.pt"


def install(remote: str = DEFAULT_REMOTE, force: bool = False) -> Path:
    from huggingface_hub import download_bucket_files, get_token, login

    if not remote.endswith(".pt") or remote.startswith("/") or ".." in Path(remote).parts:
        raise ValueError("Expected a relative .pt checkpoint path inside the bucket.")
    if DEST.is_file() and not force:
        print(f"Checkpoint already installed: {DEST}")
        print("Use --force to replace it when a new model finishes training.")
        return DEST
    if not get_token():
        print("Sign in to Hugging Face locally to read your private checkpoint.")
        login()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    temp = DEST.with_name(DEST.name + ".part")
    try:
        download_bucket_files(BUCKET, files=[(remote, str(temp))])
        if not temp.is_file() or temp.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded checkpoint is unexpectedly small.")
        temp.replace(DEST)
    finally:
        if temp.exists():
            temp.unlink()
    print(f"Model installed: {DEST} ({DEST.stat().st_size:,} bytes)")
    return DEST


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default=os.environ.get("HEXADEPTH_CHECKPOINT_REMOTE", DEFAULT_REMOTE))
    parser.add_argument("--force", action="store_true", help="Replace an existing local checkpoint")
    args = parser.parse_args()
    install(remote=args.remote, force=args.force)
