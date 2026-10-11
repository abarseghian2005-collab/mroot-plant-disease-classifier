
import json
import spaces
from pathlib import Path

import gradio as gr

import torch

import timm

from PIL import Image

from torchvision import transforms

ROOT = Path(__file__).parent

MODEL_DIR = ROOT / "models"

DEVICE = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"

)

# ---------------------------------------------------------

# Classes

# ---------------------------------------------------------

with open(ROOT / "class_mapping.json") as f:

    class_to_idx = json.load(f)

classes = sorted(

    class_to_idx,

    key=class_to_idx.get

)

NUM_CLASSES = len(classes)

def pretty_name(name):

    return name.replace("_", " ").title()

# ---------------------------------------------------------

# Model files

# ---------------------------------------------------------

checkpoint_paths = [

    MODEL_DIR / "seed1.pth",

    MODEL_DIR / "seed2.pth",

    MODEL_DIR / "seed3.pth",

]

first_checkpoint = torch.load(

    checkpoint_paths[0],

    map_location="cpu",

    weights_only=False

)

data_cfg = first_checkpoint["data_cfg"]

IMAGE_SIZE = data_cfg["input_size"][-1]

MEAN = data_cfg["mean"]

STD = data_cfg["std"]

# ---------------------------------------------------------

# Image preprocessing

# ---------------------------------------------------------

center_transform = timm.data.create_transform(

    **data_cfg,

    is_training=False

)

class ResizeAndPad:

    def __init__(self, size, mean):

        self.size = size

        self.fill = tuple(

            round(x * 255) for x in mean

        )

    def __call__(self, image):

        w, h = image.size

        scale = self.size / max(w, h)

        new_w = round(w * scale)

        new_h = round(h * scale)

        image = image.resize(

            (new_w, new_h),

            Image.Resampling.BICUBIC

        )

        canvas = Image.new(

            "RGB",

            (self.size, self.size),

            self.fill

        )

        canvas.paste(

            image,

            (

                (self.size - new_w) // 2,

                (self.size - new_h) // 2

            )

        )

        return canvas

padded_transform = transforms.Compose([

    ResizeAndPad(IMAGE_SIZE, MEAN),

    transforms.ToTensor(),

    transforms.Normalize(MEAN, STD),

])

# ---------------------------------------------------------

# Load the 3 EVA models

# ---------------------------------------------------------

models = []

for checkpoint_path in checkpoint_paths:

    checkpoint = torch.load(

        checkpoint_path,

        map_location="cpu",

        weights_only=False

    )

    model = timm.create_model(

        "eva02_small_patch14_336",

        pretrained=False,

        num_classes=NUM_CLASSES

    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    model.eval()

    model.to(DEVICE)

    models.append(model)

# ---------------------------------------------------------

# Prediction

# ---------------------------------------------------------
@spaces.GPU
@torch.inference_mode()

def predict(image):

    if image is None:

        return {}

    image = image.convert("RGB")

    center = center_transform(image)

    flipped = torch.flip(

        center,

        dims=[2]

    )

    padded = padded_transform(image)

    # Process the 3 image views together

    views = torch.stack([

        center,

        flipped,

        padded

    ]).to(DEVICE)

    predictions = []

    for model in models:

        if DEVICE.type == "cuda":

            with torch.autocast(

                device_type="cuda",

                dtype=torch.float16

            ):

                logits = model(views)

        else:

            logits = model(views)

        probs = torch.softmax(

            logits.float(),

            dim=1

        )

        normal_prob = probs[0]

        flip_prob = probs[1]

        padded_prob = probs[2]

        center_tta = (

            normal_prob + flip_prob

        ) / 2

        fixed_prediction = (

            0.75 * center_tta+ 0.25 * padded_prob

        )

        predictions.append(

            fixed_prediction

        )

    ensemble = torch.stack(

        predictions

    ).mean(dim=0)

    values, indices = torch.topk(

        ensemble,

        k=5

    )

    return {

        pretty_name(classes[index.item()]): float(value.item())

        for value, index in zip(values, indices)

    }

# ---------------------------------------------------------

# Interface

# ---------------------------------------------------------

with gr.Blocks(

    title="MROOT Plant Disease Classifier"

) as demo:

    gr.Markdown(

        """

        # 🌿 MROOT

        ## Plant Disease Classifier

        Upload a plant image and MROOT will show its

        five most likely disease classes.

        The classifier uses an ensemble of three

        EVA-02 vision models.

        """

    )

    with gr.Row():

        image_input = gr.Image(

            type="pil",

            label="Upload Plant Image"

        )

        prediction_output = gr.Label(

            num_top_classes=5,

            label="Disease Predictions"

        )

    classify = gr.Button(

        "Classify Image",

        variant="primary"

    )

    clear = gr.ClearButton(

        [

            image_input,

            prediction_output

        ]

    )

    classify.click(

        fn=predict,

        inputs=image_input,

        outputs=prediction_output

    )

if __name__ == "__main__":

    demo.launch()
