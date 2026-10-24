"""Download and checksum the two final checkpoints."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mroot.assets import ensure_model_assets

if __name__ == "__main__":
    print("Verified models:", ensure_model_assets())
