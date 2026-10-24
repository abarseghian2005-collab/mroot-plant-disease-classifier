"""MROOT V2: the frozen two-model plant disease classifier."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from threading import Lock

from PIL import Image

ROOT = Path(__file__).resolve().parent
HF_SPACE = "Armennnn/mroot-plant-disease-classifier-v2"
HF_REVISION = "cb7bf0deb1179cd2ce445bf024f355c38660fb9e"
VIEW_WEIGHTS = {"normal": 0.375, "horizontal_flip": 0.375, "whole_image_pad": 0.25}
MEMBERS = [{'name': 'clean_control_epoch7',
  'model_name': 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
  'data_cfg': {'input_size': [3, 448, 448],
               'interpolation': 'bicubic',
               'mean': [0.48145466, 0.4578275, 0.40821073],
               'std': [0.26862954, 0.26130258, 0.27577711],
               'crop_pct': 1,
               'crop_mode': 'squash'},
  'weight': 0.5,
  'weights_file': 'models/clean_control_epoch7.safetensors',
  'sha256': 'b2c230d128bbdce6a8c242dd3a5c34fc80441af530451aa6b2e94041ce2f7f3b'},
 {'name': 'clean_mixup_cutmix_epoch11',
  'model_name': 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
  'data_cfg': {'input_size': [3, 448, 448],
               'interpolation': 'bicubic',
               'mean': [0.48145466, 0.4578275, 0.40821073],
               'std': [0.26862954, 0.26130258, 0.27577711],
               'crop_pct': 1,
               'crop_mode': 'squash'},
  'weight': 0.5,
  'weights_file': 'models/clean_mixup_cutmix_epoch11.safetensors',
  'sha256': '691f3319224f0564d69aafe1aa7574b133ed35b8378d4d8d15352997d4adf7fe'}]
_PREDICTION_LOCK = Lock()


def load_classes():
    mapping = json.loads((ROOT / "class_mapping.json").read_text(encoding="utf-8"))
    if len(mapping) != 39 or sorted(mapping.values()) != list(range(39)):
        raise ValueError("The classifier requires 39 unique class indices, numbered 0 through 38.")
    return sorted(mapping, key=mapping.get)


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
    def __init__(self, device="cpu"):
        # Import model dependencies only when the first prediction is requested.
        import timm
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from torchvision import transforms

        self.torch = torch
        self.device = torch.device(device)
        self.classes = load_classes()
        self.config = {"views": VIEW_WEIGHTS}
        self.members = []
        torch.set_num_threads(max(1, int(os.environ.get("MROOT_CPU_THREADS", "2"))))
        cache = Path(os.environ.get("MROOT_MODEL_CACHE", ROOT / ".mroot_cache"))

        for member in MEMBERS:
            path = hf_hub_download(
                repo_id=HF_SPACE,
                repo_type="space",
                revision=HF_REVISION,
                filename=member["weights_file"],
                cache_dir=str(cache),
            )
            if file_sha256(path) != member["sha256"]:
                raise RuntimeError(f"Checkpoint checksum failed: {member['name']}")

            model = timm.create_model(
                member["model_name"], pretrained=False, num_classes=len(self.classes)
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

    def probabilities(self, image):
        torch = self.torch
        with torch.inference_mode():
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


@lru_cache(maxsize=1)
def get_predictor():
    return MrootEnsemble(device=os.environ.get("MROOT_DEVICE", "cpu"))


def create_demo():
    import gradio as gr

    def classify(image):
        if image is None:
            raise gr.Error("Upload a plant photo first.")
        with _PREDICTION_LOCK:
            return get_predictor().predict(image)

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Plant photo", sources=["upload"]),
        outputs=gr.Label(num_top_classes=5, label="Most likely diseases"),
        title="MROOT Plant Disease Classifier",
        description=(
            "Upload a plant photo to see five ranked disease suggestions. "
            "MROOT recognizes 39 disease categories and has no healthy or unknown category. "
            "The first prediction downloads the two saved models and can take longer."
        ),
        submit_btn="Analyze photo",
        clear_btn="Clear",
        flagging_mode="never",
        analytics_enabled=False,
        concurrency_limit=1,
        api_name="predict",
        delete_cache=(3600, 3600),
    )


if __name__ == "__main__":
    create_demo().launch()
