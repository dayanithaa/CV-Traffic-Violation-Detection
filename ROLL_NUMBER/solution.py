import os
import sys
import json
import time
import cv2
import traceback
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO

base_dir = Path(__file__).resolve().parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from utils.logger import inference_logger, timing_logger, error_logger
from utils.benchmark import Benchmark
from utils.preprocessing import TrafficImagePreprocessor
from models.detectors import BikeDetector, PoseDetector
from models.association import RiderBikeAssociator
from models.ocr import OCRProcessor
from configs.config import PATHS, MODEL_PATHS, INFERENCE

class TrafficViolationDetector:
    def __init__(self, config_override=None):
        inference_logger.info("Initializing Final TrafficViolationDetector Pipeline...")
        self.benchmark = Benchmark()
        self.fast_mode = INFERENCE.get("fast_mode", False)
        
        # Hardware optimization
        device = 'cpu' if INFERENCE.get("use_cpu", False) else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Global Image Preprocessor (Adaptive)
        self.preprocessor = TrafficImagePreprocessor()
        
        # 2. Base Detectors (Phase 4)
        self.bike_detector = BikeDetector(device=device)
        self.pose_detector = PoseDetector(device=device)
        
        # 3. Mathematical Topology Associator (Phase 5)
        self.associator = RiderBikeAssociator()
        
        # 4. Custom Fine-Tuned Classifiers (Phase 6 & 7)
        helmet_path = os.path.join(base_dir, MODEL_PATHS.get("helmet_model", "models/helmet_best.pt"))
        lp_path = os.path.join(base_dir, MODEL_PATHS.get("lp_model", "models/lp_best.pt"))
        
        self.helmet_detector = None
        if os.path.exists(helmet_path):
            self.helmet_detector = YOLO(helmet_path)
            inference_logger.info(f"Loaded Custom Helmet Model from {helmet_path}")
        else:
            inference_logger.warning(f"Helmet model not found at {helmet_path}. Helmet detection will be bypassed.")
            
        self.lp_detector = None
        if os.path.exists(lp_path):
            self.lp_detector = YOLO(lp_path)
            inference_logger.info(f"Loaded Custom License Plate Model from {lp_path}")
        else:
            inference_logger.warning(f"License Plate model not found at {lp_path}. LP bounding box logic will be bypassed.")
            
        # 5. Dual-Engine OCR & Visual Enhancer (Phase 8)
        self.ocr_processor = OCRProcessor()
        inference_logger.info("Complete 9-Stage Pipeline Loaded Successfully.")

    def _crop_image(self, img, box, margin=0.0):
        """Safely crops numpy array boundaries to prevent crashes on edge cases"""
        x1, y1, x2, y2 = map(int, box)
        h, w = img.shape[:2]
        
        if margin > 0:
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - bw * margin))
            y1 = max(0, int(y1 - bh * margin))
            x2 = min(w, int(x2 + bw * margin))
            y2 = min(h, int(y2 + bh * margin))
            
        return img[y1:y2, x1:x2]

    def predict(self, img_path):
        """
        Thread-safe, stateless inference method ensuring valid JSON is ALWAYS returned.
        """
        self.benchmark.start()
        output_json = {
            "image": os.path.basename(img_path),
            "timestamp": time.time(),
            "status": "success",
            "violations": [],
            "metadata": {}
        }

        try:
            # Stage 0: Load Image
            if not os.path.exists(img_path):
                raise ValueError(f"File not found: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("cv2.imread failed to load the image array.")
            self.benchmark.record_stage("Image Load")

            # Stage 1: Global Preprocessing (Adaptive)
            processed_img, latency_pre, stats_pre = self.preprocessor.process_adaptive(img)
            self.benchmark.record_stage("Adaptive Preprocessing")
            output_json["metadata"]["preprocessing"] = stats_pre

            # Stage 2: Base Detections (Bikes + Poses)
            bike_boxes, bike_confs = self.bike_detector.detect(processed_img)
            pose_boxes, pose_confs, keypoints = self.pose_detector.detect(processed_img)
            self.benchmark.record_stage("YOLO Base Object Detections")

            # Stage 3: Mathematical Topology Association
            associations = self.associator.associate(bike_boxes, pose_boxes, keypoints)
            self.benchmark.record_stage("Rider-Bike Topology Association")

            # Evaluate each mapped motorcycle cluster
            for assoc_idx, assoc in enumerate(associations):
                bike_box = assoc['bike']
                riders = assoc['riders']
                
                violation_entry = {
                    "bike_id": assoc_idx,
                    "bike_bbox": [float(x) for x in bike_box],
                    "riders_count": len(riders) if not assoc['inferred_rider'] else 1,
                    "helmets_detected": [],
                    "no_helmets_detected": [],
                    "is_violation": False,
                    "license_plate": None,
                    "lp_confidence": 0.0,
                    "lp_bbox": None
                }

                # Stage 4: Helmet Classification
                if self.helmet_detector is not None:
                    found_helmet = False
                    for rider_box in riders:
                        # 10% crop padding to prevent missing the top of the helmet
                        rider_crop = self._crop_image(processed_img, rider_box, margin=0.1)
                        if rider_crop.size == 0: continue
                        
                        results = self.helmet_detector(rider_crop, verbose=False)
                        boxes = results[0].boxes
                        
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            
                            if cls_id == 0: # helmet
                                violation_entry["helmets_detected"].append(conf)
                                found_helmet = True
                            elif cls_id == 1: # no_helmet
                                violation_entry["no_helmets_detected"].append(conf)
                                
                    # Core Violation Logic Check
                    if not found_helmet:
                        violation_entry["is_violation"] = True

                    # "Ghost Rider" fallback logic check
                    if assoc['inferred_rider']:
                        violation_entry["is_violation"] = True
                        violation_entry["notes"] = "Violation enforced via Inferred Rider Topology."
                        
                self.benchmark.record_stage(f"Helmet Logic Block - Bike {assoc_idx}")

                # Stage 5 & 6: License Plate Recognition + OCR
                # FAST_MODE: Skips running OCR entirely if no violation is detected, saving massive CPU time.
                if self.lp_detector is not None and (violation_entry["is_violation"] or not self.fast_mode):
                    bike_crop = self._crop_image(processed_img, bike_box, margin=0.0)
                    if bike_crop.size > 0:
                        lp_results = self.lp_detector(bike_crop, verbose=False)
                        lp_boxes = lp_results[0].boxes
                        
                        if len(lp_boxes) > 0:
                            best_lp = max(lp_boxes, key=lambda x: x.conf[0].item())
                            lp_conf = float(best_lp.conf[0].item())
                            lp_local_box = best_lp.xyxy[0].cpu().numpy()
                            
                            # 5% extra crop margin to capture full LP edges
                            lp_crop = self._crop_image(bike_crop, lp_local_box, margin=0.05)
                            
                            if lp_crop.size > 0:
                                text, ocr_conf, engine, _ = self.ocr_processor.read_plate(lp_crop)
                                
                                violation_entry["license_plate"] = text
                                violation_entry["lp_confidence"] = ocr_conf
                                violation_entry["ocr_engine"] = engine
                                
                                # Remap local box to global original image coordinates
                                bx1, by1 = bike_box[:2]
                                violation_entry["lp_bbox"] = [
                                    float(bx1 + lp_local_box[0]), 
                                    float(by1 + lp_local_box[1]), 
                                    float(bx1 + lp_local_box[2]), 
                                    float(by1 + lp_local_box[3])
                                ]
                                
                self.benchmark.record_stage(f"LP & OCR Block - Bike {assoc_idx}")

                # Output Routing
                if violation_entry["is_violation"]:
                    output_json["violations"].append(violation_entry)

            total_time = self.benchmark.get_total_latency()
            output_json["metadata"]["total_latency_ms"] = total_time * 1000

        except Exception as e:
            # Absolute crash safety guard - ALWAYS return JSON
            error_msg = traceback.format_exc()
            error_logger.error(f"Prediction Pipeline Crashed for {img_path}:\n{error_msg}")
            output_json["status"] = "error"
            output_json["error_message"] = str(e)
            
        return json.dumps(output_json, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full Traffic Violation Detection Inference Pipeline")
    parser.add_argument("--image", type=str, default="datasets/test_sample.jpg", help="Path to input image")
    args = parser.parse_args()

    # Pre-instantiate the monolith
    detector = TrafficViolationDetector()
    
    img_path = str(base_dir / args.image)
    if not os.path.exists(img_path):
        print(f"Creating placeholder test image at {img_path} to test pipeline crash-safety...")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.imwrite(img_path, dummy)

    print(f"\n--- Running Pipeline on {img_path} ---")
    
    # Run Inference
    result_json = detector.predict(img_path)
    
    print("\n================ FINAL JSON PAYLOAD ================")
    print(result_json)
    print("====================================================")
