import time
import math
import queue
import threading
from .base import BaseCheckpointWriter
from checkpoint_service.checkpoint_client_async import stage_file_copy

class YoungDalyAsyncCheckpointWriter(BaseCheckpointWriter):
    """Asynchronous implementation of Young-Daly optimum interval."""
    def __init__(
        self, 
        checkpoint_path, 
        checkpoint_times, 
        record_timing_fn, 
        remote_name,
        upload_staging_dir,
        upload_client,
        delta=60.0, 
        mttf=3600.0,
        queue_size=2
    ):
        super().__init__(checkpoint_path, checkpoint_times, record_timing_fn)
        self.remote_name = remote_name
        self.upload_staging_dir = upload_staging_dir
        self.upload_client = upload_client
        self.tau = math.sqrt(2 * delta * mttf)
        print("Young Daly Async Checkpoint Parameters:")
        print(f"- Delta: {delta}")
        print(f"- MTFF: {mttf}")
        print(f"- Tau: {self.tau}")
        
        # Async Setup
        self.tasks = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def should_save(self, elapsed_time_since_last_save, total_elapsed_time):
        """Check if fixed optimal interval has passed."""
        return (elapsed_time_since_last_save >= self.tau), 0.0

    def save_checkpoint(self, payload, epoch_idx, step_idx, phase):
        self.record_timing_fn(phase, epoch_idx, step_idx, "checkpoint_yd_async_enqueue", 0.0)
        if self.tasks.full():
            try:
                self.tasks.get_nowait()
                self.tasks.task_done()
            except queue.Empty: pass
        self.tasks.put((payload, epoch_idx, step_idx, phase))

    def _worker_loop(self):
        while not self.stop_event.is_set() or not self.tasks.empty():
            try: item = self.tasks.get(timeout=0.1)
            except queue.Empty: continue
            if item is None: break

            payload, epoch, step, phase = item
            t0 = time.time()
            self._atomic_save_checkpoint(payload)
            dt = time.time() - t0
            self.checkpoint_times.append(dt)
            self.record_timing_fn(phase, epoch, step, "checkpoint_yd_async_write", dt)

            if self.upload_client:
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