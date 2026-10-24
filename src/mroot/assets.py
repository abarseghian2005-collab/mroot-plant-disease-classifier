"""Fetch only the two selected model files from an immutable HF revision."""

import json
import os
from pathlib import Path

from .config import HF_REVISION, HF_SPACE, PROJECT_ROOT, file_sha256, load_config


def ensure_model_assets(directory=None):
    directory = Path(directory or os.environ.get("MROOT_MODEL_DIR", PROJECT_ROOT / "models"))
    directory.mkdir(parents=True, exist_ok=True)
    config = load_config()

    for member in config["members"]:
        destination = directory / member["weights_file"]
        if not destination.exists():
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=HF_SPACE,
                repo_type="space",
                revision=HF_REVISION,
                filename=member["weights_file"],
                local_dir=directory,
            )
        if file_sha256(destination) != member["sha256"]:
            raise RuntimeError(f"Checkpoint checksum failed: {destination.name}")

    metadata = directory / "ensemble.json"
    metadata.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return directory
