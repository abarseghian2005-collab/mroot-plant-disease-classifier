"""Load and validate the frozen ensemble metadata without importing PyTorch."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HF_SPACE = "Armennnn/mroot-plant-disease-classifier-v2"
HF_REVISION = "cb7bf0deb1179cd2ce445bf024f355c38660fb9e"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config):
    classes = config["classes"]
    if len(classes) != 39 or len(set(classes)) != 39:
        raise ValueError("The ensemble requires exactly 39 unique disease labels.")
    if config["class_to_idx"] != {name: index for index, name in enumerate(classes)}:
        raise ValueError("Class names and class indices disagree.")
    if config["views"] != {"normal": 0.375, "horizontal_flip": 0.375, "whole_image_pad": 0.25}:
        raise ValueError("The frozen inference view weights have changed.")
    members = config["members"]
    if len(members) != 2 or any(member["weight"] != 0.5 for member in members):
        raise ValueError("The final ensemble requires two equally weighted models.")
    if len({member["name"] for member in members}) != 2:
        raise ValueError("Ensemble members must be different checkpoints.")
    for member in members:
        raw_path = member["weights_file"]
        path = PurePosixPath(raw_path)
        if path.is_absolute() or ".." in path.parts or "\\" in raw_path or ":" in raw_path:
            raise ValueError("Weight files must stay inside the model directory.")
        if not re.fullmatch(r"[0-9a-f]{64}", member["sha256"]):
            raise ValueError("A checkpoint SHA-256 is missing or invalid.")
        data_cfg = member["data_cfg"]
        if list(data_cfg["input_size"]) != [3, 448, 448]:
            raise ValueError("The selected models require RGB images at 448 pixels.")
        if len(data_cfg["mean"]) != 3 or len(data_cfg["std"]) != 3:
            raise ValueError("RGB normalization requires three mean/std values.")
        if any(value <= 0 for value in data_cfg["std"]):
            raise ValueError("Normalization standard deviations must be positive.")
    return config


def load_config(path=None):
    path = Path(path) if path else PROJECT_ROOT / "ensemble.json"
    return validate_config(json.loads(path.read_text(encoding="utf-8")))
