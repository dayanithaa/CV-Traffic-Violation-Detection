import time
import psutil
import os
from .logger import timing_logger

class Benchmark:
    def __init__(self):
        self.start_time = None
        self.stage_times = {}
        self.process = psutil.Process(os.getpid())
        
    def start(self):
        self.start_time = time.perf_counter()
        self.stage_times.clear()
        
    def record_stage(self, stage_name):
        current_time = time.perf_counter()
        elapsed = current_time - self.start_time
        self.stage_times[stage_name] = elapsed
        self.start_time = current_time  # Reset for next stage
        
        memory_usage = self.process.memory_info().rss / (1024 * 1024) # MB
        timing_logger.info(f"Stage '{stage_name}' took {elapsed:.4f}s | CPU Mem: {memory_usage:.2f} MB")
        
        return elapsed
        
    def get_total_latency(self):
        total = sum(self.stage_times.values())
        timing_logger.info(f"Total Latency: {total:.4f}s")
        return total
