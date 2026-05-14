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
from configs.lp_data import LP_DATASET
from utils.logger import training_logger

def train_lp_model():
    # Dynamically generate hidden yaml to satisfy Ultralytics natively
    temp_yaml_path = str(base_dir / "configs" / ".temp_lp_data.yaml")
    try:
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(LP_DATASET, f, default_flow_style=False)
    except Exception as e:
        training_logger.error(f"Failed to generate dynamic YAML: {e}")
        return

    # High resolution is CRITICAL for tiny-object License Plate detection
    imgsz = TRAINING["license_plate"]["resolution"] # 960
    patience = TRAINING["license_plate"]["patience"] # 12
    
    # Tiny-Object focused Augmentations
    aug_params = {
        'mosaic': 1.0,
        'perspective': 0.002, # ~20 degrees warp for angled plates
        'hsv_v': 0.5, # Aggressive Shadow/Glare simulation
        'copy_paste': 0.2, # Extremely useful for tiny-object density
        'mixup': 0.1,
        'degrees': 10.0, # Rotation for crooked plates
        'scale': 0.5, # Scale reduction forces model to learn smaller details
    }

    try:
        # =========================================================
        # STAGE 1: Warmup & Frozen Backbone (Epochs 1-5)
        # =========================================================
        training_logger.info("Starting STAGE 1: Frozen Backbone (Epochs 1-5)")
        print("\n--- STAGE 1: WARMUP (Tiny Objects) ---")
        model = YOLO("yolov8m.pt") 
        
        model.train(
            data=temp_yaml_path,
            epochs=5,
            imgsz=imgsz,
            freeze=10, 
            lr0=0.001,
            lrf=10.0,
            patience=patience,
            project=str(base_dir / "outputs" / "lp_training"),
            name="stage1_frozen",
            **aug_params
        )
        
        last_weight_path1 = Path(model.trainer.save_dir) / "weights" / "last.pt"
        
        # =========================================================
        # STAGE 2: Unfrozen Cosine Annealing (Epochs 6-40)
        # =========================================================
        training_logger.info("Starting STAGE 2: Unfrozen Backbone (Epochs 6-40)")
        print("\n--- STAGE 2: MAIN TRAINING ---")
        model2 = YOLO(str(last_weight_path1))
        
        model2.train(
            data=temp_yaml_path,
            epochs=35, # Remaining epochs for stage 2
            imgsz=imgsz,
            freeze=0, 
            lr0=0.01,
            lrf=0.1, # Cosine drop to 0.001
            patience=patience,
            project=str(base_dir / "outputs" / "lp_training"),
            name="stage2_unfrozen",
            resume=False,
            **aug_params
        )
        
        last_weight_path2 = Path(model2.trainer.save_dir) / "weights" / "last.pt"

        # =========================================================
        # STAGE 3: Micro Fine-tuning Low LR (Epochs 41-60)
        # =========================================================
        training_logger.info("Starting STAGE 3: Fine-Tuning (Epochs 41-60)")
        print("\n--- STAGE 3: FINE TUNING ---")
        model3 = YOLO(str(last_weight_path2))
        
        model3.train(
            data=temp_yaml_path,
            epochs=20, # Remaining epochs for stage 3
            imgsz=imgsz,
            freeze=0,
            lr0=0.001,
            lrf=0.01, # Deep Cosine drop to 0.00001
            patience=patience,
            project=str(base_dir / "outputs" / "lp_training"),
            name="stage3_finetune",
            resume=False,
            **aug_params
        )
        
        # =========================================================
        # COMPLETION & ARTIFACT ROUTING
        # =========================================================
        best_weight = Path(model3.trainer.save_dir) / "weights" / "best.pt"
        final_dest = base_dir / "models" / "lp_best.pt"
        
        if best_weight.exists():
            shutil.copy(str(best_weight), str(final_dest))
            training_logger.info(f"License Plate Training Completed! Best weights copied to {final_dest}")
            print(f"\nSUCCESS: Tiny-Object Model saved to {final_dest}")
            print("Visualization plots (mAP curves, tiny-object analysis) are in the outputs/lp_training/ folder.")
        else:
            print("Warning: Could not find best.pt. Training may have halted early.")
            
    except Exception as e:
        training_logger.error(f"LP Training Pipeline Failed: {e}")
        print(f"CRITICAL ERROR: {e}")
    finally:
        # Cleanup the temporary physical yaml file
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

if __name__ == "__main__":
    train_lp_model()
