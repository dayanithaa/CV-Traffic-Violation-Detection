import argparse
from utils.reproducibility import set_seed, log_versions
from utils.logger import inference_logger

def main():
    parser = argparse.ArgumentParser(description="Traffic Violation Detection Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    set_seed(42)
    
    try:
        log_versions()
    except ImportError as e:
        inference_logger.warning(f"Could not log some versions: {e}")
        
    inference_logger.info("Starting inference pipeline...")

if __name__ == "__main__":
    main()
