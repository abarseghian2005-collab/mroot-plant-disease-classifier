"""Inference for the frozen MROOT two-model ensemble."""

import hashlib
import json
import os
from pathlib import Path

import timm
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms


class WholeImagePad:
    """Keep the whole image and pad with the checkpoint's mean RGB color."""

    def __init__(self, size, mean):
        self.size = size
        self.fill = tuple(round(value * 255) for value in mean)

    def __call__(self, image):
        width, height = image.size
        scale = self.size / max(width, height)
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
        resized = image.resize((width, height), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - width) // 2, (self.size - height) // 2))
        return canvas


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MrootEnsemble:
    def __init__(self, directory=None, device="cpu"):
        self.directory = Path(directory or Path(__file__).resolve().parents[2] / "models")
        self.device = torch.device(device)
        self.config = json.loads((self.directory / "ensemble.json").read_text())
        self.classes = self.config["classes"]
        self.members = []

        # Keep preprocessing threads bounded and process requests sequentially.
        torch.set_num_threads(max(1, int(os.environ.get("MROOT_CPU_THREADS", "2"))))

        for member in self.config["members"]:
            path = self.directory / member["weights_file"]
            if file_sha256(path) != member["sha256"]:
                raise RuntimeError(f"Checkpoint checksum failed: {path.name}")

            model = timm.create_model(
                member["model_name"],
                pretrained=False,
                num_classes=len(self.classes),
            )
            model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
            model = model.to(self.device).eval()
            model.requires_grad_(False)

            data_cfg = member["data_cfg"]
            normal = timm.data.create_transform(**data_cfg, is_training=False)
            padded = transforms.Compose([
                WholeImagePad(data_cfg["input_size"][-1], data_cfg["mean"]),
                transforms.ToTensor(),
                transforms.Normalize(data_cfg["mean"], data_cfg["std"]),
            ])
            self.members.append((model, normal, padded, member["weight"]))

    @torch.inference_mode()
    def probabilities(self, image):
        image = image.convert("RGB")
        if image.width < 1 or image.height < 1:
            raise ValueError("Please upload a valid image.")

        prediction = torch.zeros(len(self.classes), dtype=torch.float32, device=self.device)
        views = self.config["views"]

        for model, normal_transform, padded_transform, weight in self.members:
            normal = normal_transform(image).unsqueeze(0).to(self.device)
            padded = padded_transform(image).unsqueeze(0).to(self.device)

            normal_probs = model(normal).float().softmax(dim=1)[0]
            flipped_probs = model(normal.flip(dims=[3])).float().softmax(dim=1)[0]
            padded_probs = model(padded).float().softmax(dim=1)[0]

            prediction += weight * (
                views["normal"] * normal_probs
                + views["horizontal_flip"] * flipped_probs
                + views["whole_image_pad"] * padded_probs
            )

        if not torch.isfinite(prediction).all():
            raise RuntimeError("The prediction could not be calculated. Please retry.")
        return prediction.cpu()

    def predict(self, image):
        probabilities = self.probabilities(image)
        return {
            label.replace("_", " "): float(probability)
            for label, probability in zip(self.classes, probabilities)
        }
