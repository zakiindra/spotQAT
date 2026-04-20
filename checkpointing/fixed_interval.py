import time
from .base import BaseCheckpointWriter
from checkpoint_service.checkpoint_client_async import send_checkpoint_file

class FixedIntervalCheckpointWriter(BaseCheckpointWriter):
    def __init__(self, checkpoint_path, checkpoint_times, record_timing_fn, remote_name):
        super().__init__(checkpoint_path, checkpoint_times, record_timing_fn)
        self.remote_name = remote_name

    def save_checkpoint(self, payload, epoch_idx, step_idx, phase):
        t0 = time.time()
        self._atomic_save_checkpoint(payload)
        send_checkpoint_file(self.checkpoint_path, remote_name=self.remote_name)
        dt = time.time() - t0
        self.checkpoint_times.append(dt)
        self.record_timing_fn(phase, epoch_idx, step_idx, "checkpoint_fixed_interval", dt)
