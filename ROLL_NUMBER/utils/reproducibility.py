import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    """Sets the seed for reproducibility across multiple libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")

def log_versions():
    """Logs the versions of critical libraries for reproducibility."""
    import ultralytics
    import cv2
    import transformers
    import sklearn
    import albumentations
    
    versions = {
        "PyTorch": torch.__version__,
        "Ultralytics": ultralytics.__version__,
        "OpenCV": cv2.__version__,
        "Transformers": transformers.__version__,
        "Scikit-Learn": sklearn.__version__,
        "Albumentations": albumentations.__version__,
        "Numpy": np.__version__
    }
    
    for lib, ver in versions.items():
        print(f"{lib}: {ver}")
    return versions
