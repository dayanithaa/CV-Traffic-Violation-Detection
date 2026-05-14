import os
import cv2
import json
import matplotlib.pyplot as plt
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

def plot_failure_grid(results_json_path, image_dir, output_path):
    if not os.path.exists(results_json_path):
        print(f"Results JSON not found at {results_json_path}. Run evaluate_pipeline.py first!")
        return
        
    with open(results_json_path, 'r') as f:
        results = json.load(f)
        
    # Extract the first 4 processed images for qualitative side-by-side analysis
    subset = results[:4]
    if len(subset) == 0: 
        print("No results found in JSON.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()
    
    for i, res in enumerate(subset):
        ax = axes[i]
        img_path = os.path.join(image_dir, res['image'])
        
        if not os.path.exists(img_path): 
            ax.set_title("IMAGE MISSING")
            ax.axis('off')
            continue
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Plot Bounding Boxes and OCR readouts
        for viol in res.get('violations', []):
            bx1, by1, bx2, by2 = map(int, viol['bike_bbox'])
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 0, 0), 4) # Bike Box
            cv2.putText(img, "VIOLATION", (bx1, by1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            if viol.get('lp_bbox'):
                lx1, ly1, lx2, ly2 = map(int, viol['lp_bbox'])
                cv2.rectangle(img, (lx1, ly1), (lx2, ly2), (0, 255, 0), 4) # LP Box
                
                text = f"LP: {viol.get('license_plate', 'FAILED')} ({viol.get('ocr_engine')})"
                cv2.putText(img, text, (lx1, ly1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                
        ax.imshow(img)
        status = "VIOLATION CAUGHT" if len(res.get('violations', [])) > 0 else "NO VIOLATION / PASS"
        ax.set_title(f"File: {res['image']}\nStatus: {status} | Execution: {res.get('metadata',{}).get('total_latency_ms',0):.1f}ms", fontsize=14)
        ax.axis('off')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated qualitative edge-case visualization grid at {output_path}")

if __name__ == "__main__":
    results_path = str(base_dir / "outputs" / "evaluation_results.json")
    
    # Try Kaggle first, fallback to synthetic evaluation dir
    img_dir = str(base_dir / "datasets" / "raw" / "helmet" / "kaggle_helmet")
    if not os.path.exists(img_dir):
        img_dir = str(base_dir / "datasets" / "test_evaluation")
        
    out_path = str(base_dir / "visualizations" / "qualitative_failures.png")
    
    plot_failure_grid(results_path, img_dir, out_path)
