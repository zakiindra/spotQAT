import math
import time
from .base import BaseCheckpointWriter
from checkpoint_service.checkpoint_client_async import send_checkpoint_file

class YoungDalyCheckpointWriter(BaseCheckpointWriter):
    """
    Implements the Young-Daly optimum checkpoint interval for memoryless failures.
    Formula: tau = sqrt(2 * delta * MTTF)
    """
    def __init__(
        self, 
        checkpoint_path, 
        checkpoint_times, 
        record_timing_fn, 
        remote_name,
        delta=10.0,    # Default checkpoint write overhead in seconds
        mttf=3600.0    # Default Mean Time To Failure (1 hour) 
    ):
        super().__init__(checkpoint_path, checkpoint_times, record_timing_fn)
        self.remote_name = remote_name
        self.delta = delta
        self.mttf = mttf
        
        # Calculate the optimal interval (tau) immediately
        # Reference: 
        self.tau = math.sqrt(2 * self.delta * self.mttf)
        
        print(f"[Young-Daly] Initialized with delta={self.delta}s, MTTF={self.mttf}s")
        print(f"[Young-Daly] Calculated optimal checkpoint interval (tau): {self.tau:.2f} seconds")

    def should_save(self, elapsed_time_since_last_save, total_elapsed_time):
        """
        Determines if a checkpoint should be written based on the fixed optimal interval.
        Returns (triggered, risk_score). Risk score is 0.0 as it's memoryless.
        """
        triggered = elapsed_time_since_last_save >= self.tau
        
        # We return 0.0 for risk_score because Young-Daly assumes memoryless 
        # exponential failure rates where risk is constant.
        return triggered, 0.0

    def save_checkpoint(self, payload, epoch_idx, step_idx, phase):
        """
        Executes the atomic save and remote upload.
        """
        t0 = time.time()
        
        # Perform atomic write to local storage (or /dev/shm in your future setup)
        self._atomic_save_checkpoint(payload)
        
        # Synchronously send to the checkpoint server
        send_checkpoint_file(self.checkpoint_path, remote_name=self.remote_name)
        
        dt = time.time() - t0
        self.checkpoint_times.append(dt)
        
        # Log the completion with the custom Young-Daly action label
        self.record_timing_fn(
            phase, 
            epoch_idx, 
            step_idx, 
            "checkpoint_young_daly_complete", 
            dt, 
            risk_score=0.0
        )