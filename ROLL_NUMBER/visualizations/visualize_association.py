import cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from models.association import RiderBikeAssociator

def get_centroid(box):
    return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

def plot_associations(img_shape, associations, output_path):
    # Create a blank dark canvas to visualize the topology clearly
    img = np.zeros(img_shape, dtype=np.uint8)
    img[:] = (30, 30, 35) # Dark gray background
    
    # Palette to separate different bike-rider clusters visually
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]
    
    for i, assoc in enumerate(associations):
        color = colors[i % len(colors)]
        
        # 1. Draw Bike
        bx1, by1, bx2, by2 = map(int, assoc['bike'])
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 4)
        b_cent = get_centroid(assoc['bike'])
        cv2.circle(img, b_cent, 6, color, -1)
        cv2.putText(img, f"Motorcycle {i}", (bx1, by1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 2. Draw Assigned Riders & Physical Connection Lines
        for r_box in assoc['riders']:
            rx1, ry1, rx2, ry2 = map(int, r_box)
            cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
            r_cent = get_centroid(r_box)
            cv2.circle(img, r_cent, 6, (255, 255, 255), -1)
            
            # The topological connection
            cv2.line(img, r_cent, b_cent, color, 3)
            
        # Highlight if it used fallback "ghost rider" logic
        if assoc['inferred_rider']:
            cv2.putText(img, "WARNING: INFERRED RIDER", (bx1, by2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Phase 5: Rider-to-Bike Vector Association Topology\n(Primary Keypoint Logic + DBSCAN Fallback)")
    plt.axis("off")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved association topology visualization to {output_path}")

if __name__ == "__main__":
    # Simulate a dense scene with 2 bikes and 3 riders
    associator = RiderBikeAssociator()
    bike_boxes = [[100, 200, 300, 400], [400, 150, 600, 350]]
    
    # Rider 1 (on Bike 0), Rider 2 (Pillion on Bike 0), Rider 3 (on Bike 1)
    rider_boxes = [
        [150, 100, 250, 250], 
        [180, 80, 280, 220],
        [450, 50, 550, 200]
    ]
    
    # Mock skeleton hips inside the bikes
    mock_kpts = np.zeros((17, 3))
    mock_kpts[11] = [200, 220, 0.9] 
    rider_kpts = [mock_kpts, mock_kpts, mock_kpts]
    
    associations = associator.associate(bike_boxes, rider_boxes, rider_kpts)
    
    out_path = str(base_dir / "visualizations" / "association_topology.png")
    plot_associations((500, 750, 3), associations, out_path)
