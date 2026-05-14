import cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from models.ocr import OCRProcessor

def visualize_ocr(img_path, output_path):
    if not os.path.exists(img_path):
        print(f"Test image missing: {img_path}")
        return
        
    img = cv2.imread(img_path)
    if img is None: return
    
    print("Loading Phase 8 OCR Pipeline...")
    ocr = OCRProcessor()
    
    # 1. Unenhanced Direct Read (Testing what happens without Phase 8 Enhancement)
    print("Running baseline OCR on degraded image...")
    raw_text, raw_conf = ocr._run_paddle(img)
    raw_text = ocr.postprocess_text(raw_text) if raw_text else "FAILED / GARBAGE"
    
    # 2. Enhanced Full Pipeline Read (Testing the Phase 8 FSRCNN/CLAHE Logic)
    print("Running fully enhanced Pipeline...")
    final_text, conf, model_used, enhanced_img = ocr.read_plate(img)
    if not final_text: final_text = "FAILED"
    
    # Visualization setup
    plt.figure(figsize=(16, 7))
    
    # Before
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"BEFORE ENHANCEMENT (Raw Blur)\nResult: {raw_text}\nConfidence: {raw_conf:.2f}", fontsize=12)
    plt.axis("off")
    
    # After
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_img, cmap='gray')
    plt.title(f"AFTER ENHANCEMENT (Super-Res + CLAHE + Deskew)\nResult: {final_text}\nConfidence: {conf:.2f} | Engine: {model_used}", fontsize=12, color='green' if final_text != "FAILED" else 'red')
    plt.axis("off")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nSaved OCR Before/After comparison to {output_path}")

if __name__ == "__main__":
    test_img = str(base_dir / "datasets" / "test_lp.jpg")
    
    # Automatically generate a mathematically blurry dummy license plate if none exists
    if not os.path.exists(test_img):
        print("Generating a blurry/degraded dummy License Plate to test the Enhancement engine...")
        dummy = np.zeros((100, 350, 3), dtype=np.uint8)
        dummy[:] = (180, 180, 190)
        cv2.putText(dummy, "MH 12 AB 1234", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)
        
        # Aggressive Motion Blur & Noise to simulate a speeding motorcycle
        kernel = np.zeros((15, 15))
        kernel[7, :] = np.ones(15) / 15
        dummy = cv2.filter2D(dummy, -1, kernel)
        noise = np.random.normal(0, 15, dummy.shape).astype(np.uint8)
        dummy = cv2.add(dummy, noise)
        
        os.makedirs(os.path.dirname(test_img), exist_ok=True)
        cv2.imwrite(test_img, dummy)
        
    out_path = str(base_dir / "visualizations" / "ocr_comparison.png")
    visualize_ocr(test_img, out_path)
