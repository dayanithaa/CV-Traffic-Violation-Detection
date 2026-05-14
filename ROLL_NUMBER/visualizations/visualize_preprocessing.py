import cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from utils.preprocessing import TrafficImagePreprocessor

def plot_preprocessing_comparison(img_path, output_path):
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
        
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read image with cv2.")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    preprocessor = TrafficImagePreprocessor(target_size=(640, 640))
    
    print("Applying independent filters...")
    clahe_img = cv2.cvtColor(preprocessor.apply_clahe(img), cv2.COLOR_BGR2RGB)
    gamma_img = cv2.cvtColor(preprocessor.apply_gamma(img, gamma=2.0), cv2.COLOR_BGR2RGB)
    denoised_img = cv2.cvtColor(preprocessor.denoise(img), cv2.COLOR_BGR2RGB)
    norm_img = cv2.cvtColor(preprocessor.normalize_brightness(img), cv2.COLOR_BGR2RGB)
    
    print("Running adaptive pipeline...")
    adaptive_img, latency, stats = preprocessor.process_adaptive(img)
    adaptive_img_rgb = cv2.cvtColor(adaptive_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image\n(Before Processing)")
    plt.axis("off")
    
    plt.subplot(2, 3, 2)
    plt.imshow(clahe_img)
    plt.title("CLAHE\n(Local Contrast Enhancement)")
    plt.axis("off")
    
    plt.subplot(2, 3, 3)
    plt.imshow(denoised_img)
    plt.title("Denoised\n(Non-Local Means)")
    plt.axis("off")
    
    plt.subplot(2, 3, 4)
    plt.imshow(gamma_img)
    plt.title("Gamma Corrected\n(Shadow Brightening)")
    plt.axis("off")
    
    plt.subplot(2, 3, 5)
    plt.imshow(norm_img)
    plt.title("Brightness Normalized\n(Min-Max Scaling)")
    plt.axis("off")
    
    plt.subplot(2, 3, 6)
    plt.imshow(adaptive_img_rgb)
    plt.title(f"Final Adaptive Output (Resized)\nDark: {stats['dark']} | Noisy: {stats['noisy']} | Latency: {latency:.1f}ms")
    plt.axis("off")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved preprocessing visualization to {output_path}")

if __name__ == "__main__":
    # If no image exists to test on, we generate a highly-degraded dummy image
    test_img_path = str(base_dir / "datasets" / "test_sample.jpg")
    if not os.path.exists(test_img_path):
        print("Generating a degraded dummy image for testing...")
        os.makedirs(os.path.dirname(test_img_path), exist_ok=True)
        # Create a dark, noisy image
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy[:] = (30, 40, 35) # Dark baseline
        noise = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.randn(noise, (0, 0, 0), (50, 50, 50)) # Add noise
        dummy = cv2.add(dummy, noise)
        cv2.imwrite(test_img_path, dummy)
        
    out_path = str(base_dir / "visualizations" / "preprocessing_comparison.png")
    plot_preprocessing_comparison(test_img_path, out_path)
