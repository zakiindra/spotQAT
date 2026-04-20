import os
import time
import queue
import threading
from .base import BaseCheckpointWriter
from checkpoint_service.checkpoint_client_async import stage_file_copy

class AsyncCheckpointWriter(BaseCheckpointWriter):
    def __init__(
        self,
        checkpoint_path,
        upload_staging_dir,
        checkpoint_times,
        record_timing_fn,
        upload_client,
        queue_size=2,
        remote_name="latest_spot_checkpoint.pt"
    ):
        super().__init__(checkpoint_path, checkpoint_times, record_timing_fn)
        self.upload_staging_dir = upload_staging_dir
        self.upload_client = upload_client
        self.remote_name = remote_name
        self.tasks = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _worker_loop(self):
        while not self.stop_event.is_set() or not self.tasks.empty():
            try:
                item = self.tasks.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self.tasks.task_done()
                break

            payload, epoch_idx, step_idx, phase = item
            t0 = time.time()
            self._atomic_save_checkpoint(payload)
            dt = time.time() - t0
            self.checkpoint_times.append(dt)
            self.record_timing_fn(phase, epoch_idx, step_idx, "checkpoint_async_write", dt)

            if self.upload_client is not None:
                stage_name = f"upload_{phase}_epoch{epoch_idx}_step{step_idx}.pt"
                staged_path = stage_file_copy(
                    self.checkpoint_path,
                    self.upload_staging_dir,
                    staged_name=stage_name,
                )
                self.upload_client.enqueue_file(
                    file_path=staged_path,
                    remote_name=self.remote_name,
                    delete_after=True,
                )
                self.record_timing_fn(phase, epoch_idx, step_idx, "checkpoint_async_upload_enqueue", 0.0)

            self.tasks.task_done()

    def save_checkpoint(self, payload, epoch_idx, step_idx, phase):
        t0 = time.time()
        enqueue_dt = time.time() - t0
        self.checkpoint_times.append(enqueue_dt)
        self.record_timing_fn(phase, epoch_idx, step_idx, "checkpoint_async_enqueue", enqueue_dt)
        self._enqueue(payload, epoch_idx, step_idx, phase)

    def _enqueue(self, payload, epoch_idx, step_idx, phase):
        if self.tasks.full():
            try:
                dropped = self.tasks.get_nowait()
                if dropped is not None:
                    _, drop_epoch, drop_step, drop_phase = dropped
                    self.record_timing_fn(
                        drop_phase,
                        drop_epoch,
                        drop_step,
                        "checkpoint_async_drop",
                        0.0,
                    )
                self.tasks.task_done()
            except queue.Empty:
                pass
        self.tasks.put((payload, epoch_idx, step_idx, phase))

    def flush(self):
        self.tasks.join()

    def close(self):
        self.flush()
        self.stop_event.set()
        self.tasks.put(None)
        self.worker.join()
