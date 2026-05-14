import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset Configuration defined purely in Python
HELMET_DATASET = {
    "path": str(BASE_DIR / "datasets" / "processed" / "helmet"),
    "train": "images/train",
    "val": "images/val",
    "names": {
        0: "helmet",
        1: "no_helmet"
    }
}
