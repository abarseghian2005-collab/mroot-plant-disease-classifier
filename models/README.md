# Model files

Run `python scripts/download_models.py` from the repository root. The script
downloads the two selected safetensors files from the verified, immutable
Hugging Face Space revision and checks their SHA-256 values.

Downloads total about 691 MB. The model files belong in `models/models/`
because the public Space stores them under a `models/` directory. The loader
preserves that relative layout deliberately.

No training data or Hugging Face token is needed for inference. Model weights
stay outside Git. The source checkpoints remain in the owner's private
Kaggle backup, which contains all eight selected historical/final models.
