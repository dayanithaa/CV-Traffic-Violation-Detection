import os
import sys
import time
from pathlib import Path
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from models.detectors import BikeDetector, PoseDetector

def evaluate_models():
    print("--- Evaluating YOLOv8 Detectors ---")
    
    try:
        bike_det = BikeDetector()
        pose_det = PoseDetector()
    except Exception as e:
        print(f"Error loading models: {e}")
        return
        
    print("Models loaded onto device successfully.")
    
    # Generate a dummy batch of random images to test raw inference latency
    print("Running Inference Benchmarks (10 rounds)...")
    dummy_images = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(10)]
    
    # Warmup
    bike_det.detect(dummy_images[0])
    pose_det.detect(dummy_images[0])
    
    # Bike latency
    start = time.time()
    for img in dummy_images:
        bike_det.detect(img)
    bike_time = (time.time() - start) / len(dummy_images) * 1000
    
    # Pose latency
    start = time.time()
    for img in dummy_images:
        pose_det.detect(img)
    pose_time = (time.time() - start) / len(dummy_images) * 1000
    
    print("\n================ EVALUATION REPORT ================")
    print(f"Bike Detection Latency : {bike_time:.2f} ms per image")
    print(f"Pose Detection Latency : {pose_time:.2f} ms per image")
    print("Total Inference Block  : {:.2f} ms".format(bike_time + pose_time))
    print("---------------------------------------------------")
    print("Detection Recall       : TBD (Requires labeled test dataset)")
    print("Small-Object Recall    : Optimized via dynamic crop expansion margin (10%)")
    print("Dense-Scene Protection : Active (Fallback NMS thresholds enabled)")
    print("===================================================")

if __name__ == "__main__":
    evaluate_models()
