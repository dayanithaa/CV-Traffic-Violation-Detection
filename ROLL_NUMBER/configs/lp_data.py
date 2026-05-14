import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset Configuration defined purely in Python
LP_DATASET = {
    "path": str(BASE_DIR / "datasets" / "processed" / "lp"),
    "train": "images/train",
    "val": "images/val",
    "names": {
        0: "license_plate"
    }
}
