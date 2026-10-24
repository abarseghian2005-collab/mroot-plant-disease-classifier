"""Validate input images and training metadata before model execution."""

import csv
import io
from pathlib import Path
import warnings

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImage(ValueError):
    pass


def decode_image(data):
    if not data:
        raise InvalidImage("Upload an image first.")
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidImage("The image must be smaller than 20 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in ALLOWED_FORMATS:
                    raise InvalidImage("Use a JPEG, PNG, or WebP image.")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise InvalidImage("The image must contain at most 25 million pixels.")
                image.load()
                return image.convert("RGB")
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise InvalidImage("The uploaded file could not be read as an image.") from error


class ManifestRow(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str = Field(min_length=1)
    disease: str = Field(min_length=1)
    label: int = Field(ge=0, lt=39)
    duplicate_group: str = Field(min_length=1)


def validate_manifest(path, mapping, expected_rows=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [ManifestRow.model_validate(row) for row in csv.DictReader(stream)]
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(rows)}.")
    if not rows or {row.label for row in rows} != set(range(39)):
        raise ValueError("The manifest must include every disease class.")
    for row in rows:
        if mapping.get(row.disease) != row.label:
            raise ValueError(f"Class mapping mismatch for {row.disease}.")
    return rows


def validate_training_splits(train_rows, validation_rows):
    train_paths = {row.path for row in train_rows}
    validation_paths = {row.path for row in validation_rows}
    train_groups = {row.duplicate_group for row in train_rows}
    validation_groups = {row.duplicate_group for row in validation_rows}
    if train_paths & validation_paths:
        raise ValueError("Training and validation share image paths.")
    if train_groups & validation_groups:
        raise ValueError("Training and validation share detected image groups.")
