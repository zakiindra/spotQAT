import os
import time
import json
import queue
import threading
import numpy as np
import pandas as pd
from .base import BaseCheckpointWriter
from checkpoint_service.checkpoint_client_async import stage_file_copy

class KaplanMeierAsyncCheckpointWriter(BaseCheckpointWriter):
    def __init__(
        self, 
        checkpoint_path, 
        checkpoint_times, 
        record_timing_fn, 
        remote_name,
        upload_staging_dir,
        upload_client,
        data_source="gcp", 
        risk_threshold=0.05, 
        window_size=600, 
        max_sample_time=float('inf'),
        min_interval=300,
        scale_factor=1.0,  # Added to support compressed time experiments
        queue_size=2
    ):
        super().__init__(checkpoint_path, checkpoint_times, record_timing_fn)
        self.remote_name = remote_name
        self.upload_staging_dir = upload_staging_dir
        self.upload_client = upload_client
        self.data_source = data_source
        self.risk_threshold = risk_threshold
        self.max_sample_time = max_sample_time
        self.min_interval = min_interval
        self.scale_factor = scale_factor
        
        # Scale window_size proportionately if max_sample_time is constrained
        self.window_size = min(window_size, max_sample_time * 0.1) if max_sample_time != float('inf') else window_size
        
        # Async Task Queue Setup 
        self.tasks = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        
        # Load and Scale Risk Data
        self.lifetimes = self._load_data()
        self.km_survival = self._compute_kaplan_meier()
        print("Kaplan Meier Async Checkpoint Parameters:")
        print(f"- Risk threshold: {self.risk_threshold}")
        print(f"- Window size: {self.window_size}")
        print(f"- Min interval: {self.min_interval}")
        print(f"- Scale factor: {self.scale_factor}")   
        
        self.worker.start()

    def _load_data(self):
        """Loads and scales empirical trace data."""
        lifetimes = []
        if self.data_source == "gcp":
            data_path = os.path.join(os.path.dirname(__file__), "..", "data", "gcp", "data.json")
            if os.path.exists(data_path):
                with open(data_path, "r") as f:
                    data = json.load(f)
                lifetimes = [val["time_in_sec"] for val in data.values() if "time_in_sec" in val]
        elif self.data_source == "aws":
            data_path = os.path.join(os.path.dirname(__file__), "..", "data", "aws", "us-east-1a_cdf.csv")
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                if 'Duration' in df.columns: lifetimes = df['Duration'].tolist()
        
        # APPLY SCALE FACTOR: Compresses 24h risk into experimental window
        lifetimes = [t / self.scale_factor for t in lifetimes]
        
        # Filter by the now-scaled max_sample_time
        lifetimes = [t for t in lifetimes if t <= self.max_sample_time]
        return np.sort(lifetimes)

    def _compute_kaplan_meier(self):
        """Standard KM estimate for survival probability."""
        n = len(self.lifetimes)
        if n == 0: return lambda t: 1.0
        survival = 1.0 - (np.arange(1, n + 1) / n)
        def survival_fn(t):
            idx = np.searchsorted(self.lifetimes, t, side='right')
            return survival[idx-1] if 0 < idx < n else (1.0 if idx == 0 else 0.0)
        return survival_fn

    def get_conditional_survival(self, current_time, window):
        """Probability of surviving until current_time + window given survival until now."""
        s_current = self.km_survival(current_time)
        if s_current <= 0: return 0.0
        return self.km_survival(current_time + window) / s_current

    def should_save(self, elapsed_time_since_last_save, total_elapsed_time):
        """Evaluates bathtub risk. Returns (triggered, risk_score)."""
        survival_prob = self.get_conditional_survival(total_elapsed_time, self.window_size)
        failure_prob = 1.0 - survival_prob
        
        # Trigger if risk exceeds threshold after min_interval, or after 1h backup interval
        triggered = (failure_prob > self.risk_threshold and elapsed_time_since_last_save > self.min_interval) or \
                    (elapsed_time_since_last_save > 3600)
        return triggered, failure_prob

    def save_checkpoint(self, payload, epoch_idx, step_idx, phase):
        """Enqueues payload for background worker."""
        self.record_timing_fn(phase, epoch_idx, step_idx, "checkpoint_adaptive_async_enqueue", 0.0)
        if self.tasks.full():
            try:
                self.tasks.get_nowait()
                self.tasks.task_done()
            except queue.Empty: pass
        self.tasks.put((payload, epoch_idx, step_idx, phase))

    def _worker_loop(self):
        """Handles heavy I/O in parallel with training steps."""
        while not self.stop_event.is_set() or not self.tasks.empty():
            try: item = self.tasks.get(timeout=0.1)
            except queue.Empty: continue
            if item is None: break

            payload, epoch, step, phase = item
            t0 = time.time()
            self._atomic_save_checkpoint(payload)
            dt = time.time() - t0
            self.checkpoint_times.append(dt)
            self.record_timing_fn(phase, epoch, step, "checkpoint_adaptive_async_write", dt)

            if self.upload_client:
                # Capture staging write time for future I/O simulation statistics
                staged_path, copy_dt = stage_file_copy(self.checkpoint_path, self.upload_staging_dir)
                self.record_timing_fn(phase, epoch, step, "checkpoint_async_staging_write", copy_dt)
                self.upload_client.enqueue_file(staged_path, self.remote_name, delete_after=True)
            self.tasks.task_done()

    def flush(self): self.tasks.join()
    def close(self):
        self.flush()
        self.stop_event.set()
        self.tasks.put(None)
        self.worker.join()