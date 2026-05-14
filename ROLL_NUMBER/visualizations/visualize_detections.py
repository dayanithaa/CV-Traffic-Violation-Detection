import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from models.detectors import BikeDetector, PoseDetector

def draw_pose(img, keypoints, bbox=None):
    img_draw = img.copy()
    
    # COCO pose skeleton connections
    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], 
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], 
                [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], 
                [3, 5], [4, 6]]
                
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
    for k in keypoints:
        # Draw joints
        for point in k:
            x, y, conf = point
            if conf > 0:
                cv2.circle(img_draw, (int(x), int(y)), 4, (0, 255, 0), -1)
                
        # Draw skeleton lines
        for c1, c2 in skeleton:
            if k[c1][2] > 0 and k[c2][2] > 0:
                pt1 = (int(k[c1][0]), int(k[c1][1]))
                pt2 = (int(k[c2][0]), int(k[c2][1]))
                cv2.line(img_draw, pt1, pt2, (0, 255, 255), 2)
                
    return img_draw

def create_confidence_heatmap(img_shape, boxes, confs):
    heatmap = np.zeros(img_shape[:2], dtype=np.float32)
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)
        heatmap[y1:y2, x1:x2] += conf
        
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return heatmap_colored

def plot_detection_results(img_path, output_path):
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
        
    img = cv2.imread(img_path)
    
    print("Initializing Detectors (Downloading weights if first time)...")
    bike_detector = BikeDetector()
    pose_detector = PoseDetector()
    
    print("Running inference...")
    bike_boxes, bike_confs = bike_detector.detect(img)
    pose_boxes, pose_confs, keypoints = pose_detector.detect(img)
    
    bike_overlay = img.copy()
    for box, conf in zip(bike_boxes, bike_confs):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(bike_overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(bike_overlay, f"Bike {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    pose_overlay = draw_pose(img, keypoints)
    heatmap = create_confidence_heatmap(img.shape, bike_boxes, bike_confs)
    
    combined = img.copy()
    for box in bike_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 0, 255), 2)
    combined = draw_pose(combined, keypoints)
    
    plt.figure(figsize=(16, 16))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(bike_overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Motorcycle Detection ({len(bike_boxes)} found)")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(pose_overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Rider Pose Detection ({len(pose_boxes)} found)")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.title("Detection Density Heatmap")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title("Combined Dense Scene")
    plt.axis("off")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    test_img_path = str(base_dir / "datasets" / "test_sample.jpg")
    out_path = str(base_dir / "visualizations" / "detection_comparison.png")
    plot_detection_results(test_img_path, out_path)
