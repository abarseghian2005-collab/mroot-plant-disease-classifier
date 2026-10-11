# MROOT Plant Disease Classifier

MROOT is a computer vision project for classifying plant diseases from images.

The system predicts one of *39 disease classes*, independent of plant species.

## Current Model

The deployed system uses an ensemble of *3 EVA-02 Small models* at 336×336 resolution.

Each model evaluates 3 views of the image:

- normal image

- horizontally flipped image

- full-image padded view

The predictions from the three views are combined, then the outputs of the three independently trained models are averaged.

## Results

- Clean ViT-Small baseline macro mAP: *0.9064*

- 2-model EVA ensemble macro mAP: *0.9353*

- Final 3-model EVA ensemble macro mAP: *0.9472*

Dataset split after duplicate/conflict cleanup:

- Training: *6,870 images*

- Validation: *390 images*

- Final holdout: *670 images*

- Classes: *39*

The final holdout was kept separate from model selection.

## Deployment

The model is deployed with Gradio on Hugging Face Spaces.

*Live demo:* https://huggingface.co/spaces/Armennnn/mroot-plant-disease-classifier

## Repository Files

- app.py — inference and Gradio interface

- requirements.txt — Python dependencies

- class_mapping.json — mapping for the 39 disease classes

Model checkpoints are stored on Hugging Face rather than directly in this GitHub repository.

## Tech Stack

- Python

- PyTorch

- timm

- EVA-02

- Gradio

- Hugging Face Spaces

- Kaggle
