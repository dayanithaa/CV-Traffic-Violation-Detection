import os
import sys
import shutil
import yaml
from pathlib import Path

# Add project root to path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from ultralytics import YOLO
from configs.config import TRAINING
from configs.helmet_data import HELMET_DATASET
from utils.logger import training_logger

def train_helmet_model():
    # Ultralytics YOLOv8 strictly requires a physical .yaml file for dataset paths under the hood.
    # To keep our architecture pure-Python, we dynamically generate a hidden temporary yaml 
    # from our Python dictionary, use it for training, and then immediately delete it.
    temp_yaml_path = str(base_dir / "configs" / ".temp_helmet_data.yaml")
    try:
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(HELMET_DATASET, f, default_flow_style=False)
    except Exception as e:
        training_logger.error(f"Failed to generate dynamic YAML: {e}")
        return

    # Pull variables from Central Config
    imgsz = TRAINING["helmet"]["resolution"] # 640
    patience = TRAINING["helmet"]["patience"] # 10
    
    aug_params = {
        'mosaic': 1.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'perspective': 0.001,
        'fliplr': 0.5,
        'erasing': 0.3,
    }

    try:
        # =========================================================
        # STAGE 1: Warmup & Frozen Backbone (Epochs 1-5)
        # =========================================================
        training_logger.info("Starting STAGE 1: Frozen Backbone (Epochs 1-5)")
        print("\n--- STAGE 1: WARMUP ---")
        model = YOLO("yolov8m.pt") 
        
        model.train(
            data=temp_yaml_path,
            epochs=5,
            imgsz=imgsz,
            freeze=10, 
            lr0=0.001,
            lrf=10.0,
            patience=patience,
            project=str(base_dir / "outputs" / "helmet_training"),
            name="stage1_frozen",
            **aug_params
        )
        
        last_weight_path1 = Path(model.trainer.save_dir) / "weights" / "last.pt"
        
        # =========================================================
        # STAGE 2: Unfrozen Cosine Annealing (Epochs 6-35)
        # =========================================================
        training_logger.info("Starting STAGE 2: Unfrozen Backbone (Epochs 6-35)")
        print("\n--- STAGE 2: MAIN TRAINING ---")
        model2 = YOLO(str(last_weight_path1))
        
        model2.train(
            data=temp_yaml_path,
            epochs=30,
            imgsz=imgsz,
            freeze=0, 
            lr0=0.01,
            lrf=0.1, 
            patience=patience,
            project=str(base_dir / "outputs" / "helmet_training"),
            name="stage2_unfrozen",
            resume=False,
            **aug_params
        )
        
        last_weight_path2 = Path(model2.trainer.save_dir) / "weights" / "last.pt"

        # =========================================================
        # STAGE 3: Fine-tuning Low LR (Epochs 36-50)
        # =========================================================
        training_logger.info("Starting STAGE 3: Fine-Tuning (Epochs 36-50)")
        print("\n--- STAGE 3: FINE TUNING ---")
        model3 = YOLO(str(last_weight_path2))
        
        model3.train(
            data=temp_yaml_path,
            epochs=15,
            imgsz=imgsz,
            freeze=0,
            lr0=0.001,
            lrf=0.1, 
            patience=patience,
            project=str(base_dir / "outputs" / "helmet_training"),
            name="stage3_finetune",
            resume=False,
            **aug_params
        )
        
        # =========================================================
        # COMPLETION & ARTIFACT ROUTING
        # =========================================================
        best_weight = Path(model3.trainer.save_dir) / "weights" / "best.pt"
        final_dest = base_dir / "models" / "helmet_best.pt"
        
        if best_weight.exists():
            shutil.copy(str(best_weight), str(final_dest))
            training_logger.info(f"Helmet Training Completed successfully! Best weights copied to {final_dest}")
            print(f"\nSUCCESS: Model saved to {final_dest}")
        else:
            print("Warning: Could not find best.pt. Training may have halted early.")
            
    except Exception as e:
        training_logger.error(f"Helmet Training Pipeline Failed: {e}")
        print(f"CRITICAL ERROR: {e}")
    finally:
        # Cleanup the temporary physical yaml file
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

if __name__ == "__main__":
    train_helmet_model()
