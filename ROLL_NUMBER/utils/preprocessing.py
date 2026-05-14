import cv2
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path for logging import
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from utils.logger import timing_logger

class TrafficImagePreprocessor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
    def is_dark_scene(self, img, threshold=80):
        """Detect if the scene is too dark by calculating average pixel intensity."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return np.mean(gray) < threshold
        
    def is_noisy_scene(self, img, threshold=20):
        """Detect noise by measuring the variance of the Laplacian."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold
        
    def apply_clahe(self, img):
        """Contrast Limited Adaptive Histogram Equalization."""
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = self.clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return self.clahe.apply(img)
        
    def apply_gamma(self, img, gamma=1.5):
        """Gamma correction to brighten shadows without blowing out highlights."""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
        
    def normalize_brightness(self, img):
        """Global Min-Max brightness normalization."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.HSV2BGR)
        
    def denoise(self, img):
        """Non-local means denoising (fast algorithm)."""
        return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
        
    def resize_preserve_aspect(self, img):
        """Resizes image to target size while preserving aspect ratio by padding."""
        h, w = img.shape[:2]
        tw, th = self.target_size
        
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad remaining space
        top = (th - new_h) // 2
        bottom = th - new_h - top
        left = (tw - new_w) // 2
        right = tw - new_w - left
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, scale, top, left

    def process_adaptive(self, img):
        """
        Main pipeline method: analyzes image and conditionally applies preprocessing.
        Returns the processed image, latency (ms), and metadata.
        """
        start_t = time.perf_counter()
        
        dark = self.is_dark_scene(img)
        noisy = self.is_noisy_scene(img)
        
        processed = img.copy()
        
        # Apply conditional enhancements
        if dark:
            processed = self.apply_gamma(processed, gamma=1.5)
            processed = self.apply_clahe(processed)
            
        if noisy:
            processed = self.denoise(processed)
            
        # Optional: always ensure a base level of brightness
        # processed = self.normalize_brightness(processed)
        
        # Final aspect-preserving resize for YOLO
        resized, scale, pad_t, pad_l = self.resize_preserve_aspect(processed)
        
        latency_ms = (time.perf_counter() - start_t) * 1000
        timing_logger.info(f"Adaptive Preprocessing latency: {latency_ms:.2f}ms (Dark={dark}, Noisy={noisy})")
        
        return resized, latency_ms, {"dark": dark, "noisy": noisy, "scale": scale, "pad": (pad_t, pad_l)}
