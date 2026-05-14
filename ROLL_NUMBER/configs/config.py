import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

PATHS = {
    "models": os.path.join(BASE_DIR, "models"),
    "datasets": os.path.join(BASE_DIR, "datasets"),
    "outputs": os.path.join(BASE_DIR, "outputs"),
    "logs": os.path.join(BASE_DIR, "outputs", "logs")
}

MODEL_PATHS = {
    "yolov8_bike": "yolov8s.pt",
    "yolov8_pose": "yolov8s-pose.pt",
    "helmet_model": os.path.join(PATHS["models"], "helmet_best.pt"),
    "lp_model": os.path.join(PATHS["models"], "lp_best.pt")
}

THRESHOLDS = {
    "bike_conf_primary": 0.4,
    "bike_conf_fallback": 0.25,
    "nms_iou": 0.5,
    "keypoint_conf": 0.3,
    "rider_bike_iou_fallback": 0.15,
    "ocr_char_conf_min": 0.6
}

AUGMENTATIONS = {
    "mosaic": True,
    "hsv_jitter": True,
    "perspective": 15,
    "horizontal_flip": 0.5,
    "cutout": 0.3,
    "gamma_darkening": True,
    "motion_blur": True,
    "jpeg_compression": True,
    "rain_fog": True
}

TRAINING = {
    "helmet": {
        "epochs": 50,
        "resolution": 640,
        "patience": 10
    },
    "license_plate": {
        "epochs": 60,
        "resolution": 960,
        "patience": 12
    }
}

INFERENCE = {
    "fast_mode": False,
    "use_cpu": False,
    "enable_fsrcnn": True
}
