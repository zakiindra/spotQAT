import os
import torch

class BaseCheckpointWriter:
    def __init__(self, checkpoint_path, checkpoint_times, record_timing_fn):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_times = checkpoint_times
        self.record_timing_fn = record_timing_fn

    def _atomic_save_checkpoint(self, payload):
        temp_path = self.checkpoint_path + ".tmp"
        torch.save(payload, temp_path)
        os.replace(temp_path, self.checkpoint_path)

    def save_checkpoint(self, payload, epoch_idx, step_idx, phase):
        raise NotImplementedError

    def flush(self):
        pass

    def close(self):
        pass
