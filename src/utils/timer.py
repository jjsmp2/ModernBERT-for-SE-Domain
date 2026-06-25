"""
Timer utilities for performance monitoring
"""

import time
import psutil
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

class Timer:
    """Context manager for timing operations"""

    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.time()
        if self.logger:
            self.logger.info(f"⏱️  Starting: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

        if exc_type is None:
            if self.logger:
                self.logger.info(f"✅ Completed: {self.operation_name} in {self.get_elapsed_formatted()}")
        else:
            if self.logger:
                self.logger.error(f"❌ Failed: {self.operation_name} after {self.get_elapsed_formatted()}")

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.elapsed_time is not None:
            return self.elapsed_time
        elif self.start_time is not None:
            return time.time() - self.start_time
        else:
            return 0.0

    def get_elapsed_formatted(self) -> str:
        """Get formatted elapsed time"""
        elapsed = self.get_elapsed()

        if elapsed < 60:
            return f"{elapsed:.2f} seconds"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = elapsed % 60
            return f"{hours}h {minutes}m {seconds:.1f}s"

class PerformanceMonitor:
    """Monitor system performance during operations"""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return self.process.cpu_percent()
        except:
            return 0.0

    def update_peak_memory(self):
        """Update peak memory usage"""
        current_memory = self.get_memory_usage()
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        current_memory = self.get_memory_usage()
        self.update_peak_memory()

        return {
            'initial_memory_mb': self.initial_memory,
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': current_memory - self.initial_memory,
            'cpu_usage_percent': self.get_cpu_usage()
        }
