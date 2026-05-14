import logging
import os
from datetime import datetime

def setup_logger(name, log_dir="outputs/logs", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

# Pre-configured loggers
training_logger = setup_logger("training")
inference_logger = setup_logger("inference")
error_logger = setup_logger("error", level=logging.ERROR)
timing_logger = setup_logger("timing")
