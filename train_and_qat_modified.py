import argparse
import math
import os
import csv
import time
import statistics
import copy
import queue
import threading
import shutil
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
import bitsandbytes as bnb
import gc
from prettytable import PrettyTable
from tqdm import tqdm

# TorchAO QAT
from torchao.quantization import quantize_, Int8DynamicActivationIntxWeightConfig, PerGroup
from torchao.quantization.qat import QATConfig

from checkpoint_service.checkpoint_client_async import (
    AsyncCheckpointClient,
    download_checkpoint_file,
)
from checkpointing import FixedIntervalCheckpointWriter, AsyncCheckpointWriter, KaplanMeierCheckpointWriter

# Configure Envs
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HOME"] = "./models"

# -----------------------------
# Configuration
# -----------------------------
MODELS = [
    "meta-llama/Llama-3.2-1B",
    # "meta-llama/Llama-3.2-3B",
]

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SEQ_LEN = 256
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
GRAD_ACCUM = 8
LR = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS_FP = 3
NUM_EPOCHS_QAT = 3
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0

SAVE_EVERY_N_STEPS = 200
SAVE_EVERY_N_SECONDS = -1

#   1) "fixed_interval" -> save on a fixed step/time interval in the training thread
#                           and synchronously upload to the checkpoint server
#   2) "async"          -> save on a fixed step/time interval in a background thread,
#                           then upload in a second background thread
#   3) "adaptive"       -> save based on Kaplan-Meier preemption risk analysis
CHECKPOINT_MODE = "none"  # options: "fixed_interval", "async", "adaptive"
ASYNC_CHECKPOINT_QUEUE_SIZE = 2
ASYNC_UPLOAD_QUEUE_SIZE = 2

SEED = 42

BASE_OUTPUT_DIR = "./qat_experiment_out"
BASE_CHECKPOINT_DIR = "./spot_checkpoints"
REMOTE_CHECKPOINT_FILENAME = "latest_spot_checkpoint.pt"





def _to_cpu_state_dict(state_dict):
    cpu_state = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = copy.deepcopy(value)
    return cpu_state


def run_training_pipeline(model_name, run_type):
    """
    run_type: "spot" or "baseline"
    Returns timing statistics for the run.
    """
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM)
    aprint = accelerator.print
    
    aprint(f"\n{'=' * 50}")
    aprint(f"Starting pipeline for model: {model_name} | run_type: {run_type}")
    aprint(f"{'=' * 50}\n")

    torch.manual_seed(SEED)

    DEVICE = accelerator.device
    DTYPE = (
        torch.bfloat16
        if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float32
    )

    model_short_name = model_name.split("/")[-1]

    suffix = "_spot" if run_type == "spot" else "_baseline"
    output_dir = os.path.join(BASE_OUTPUT_DIR, model_short_name + suffix)
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_DIR, model_short_name + suffix)
    upload_staging_dir = os.path.join(checkpoint_dir, "upload_staging")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(upload_staging_dir, exist_ok=True)

    # -----------------------------
    # Data prep
    # -----------------------------
    aprint(f"Loading dataset and tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(DATASET_NAME, DATASET_CONFIG)

    def tokenize_fn(batch):
        texts = [t for t in batch["text"] if t and not t.isspace()]
        return tokenizer(texts, truncation=False)

    tokenized = raw.map(
        tokenize_fn, batched=True, remove_columns=raw["train"].column_names
    )

    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total_length = (len(concatenated) // SEQ_LEN) * SEQ_LEN
        concatenated = concatenated[:total_length]
        input_ids = [
            concatenated[i : i + SEQ_LEN] for i in range(0, total_length, SEQ_LEN)
        ]
        attention_mask = [[1] * SEQ_LEN for _ in range(len(input_ids))]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [x[:] for x in input_ids],
        }

    lm_ds = tokenized.map(group_texts, batched=True)
    train_loader = DataLoader(
        lm_ds["train"],
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    eval_loader = DataLoader(
        lm_ds["validation"],
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    # -----------------------------
    # Model
    # -----------------------------
    aprint(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=DTYPE,
        device_map={"": accelerator.process_index}
    )
    model.config.use_cache = False
    
    # -----------------------------
    # Optimizer / scheduler
    # -----------------------------
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
    total_steps = (NUM_EPOCHS_FP + NUM_EPOCHS_QAT) * num_update_steps_per_epoch
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    def record_timing(phase, epoch, step, action, duration, risk_score=0.0):
        """
        Logs events with microsecond precision timestamps.
        """
        csv_path = os.path.join(BASE_OUTPUT_DIR, "timing_log.csv")
        file_exists = os.path.isfile(csv_path)
        
        # Capture absolute timestamp at the moment of logging
        current_ts = time.time() 

        with open(csv_path, mode="a", newline="") as f:
            # Added 'timestamp' and 'risk_score' to fieldnames 
            writer = csv.DictWriter(
                f,
                fieldnames=["timestamp", "model_name", "run_type", "phase", "epoch", "step", "action", "duration", "risk_score"],
            )
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                "timestamp": f"{current_ts:.6f}",
                "model_name": model_name,
                "run_type": run_type,
                "phase": phase,
                "epoch": epoch,
                "step": step,
                "action": action,
                "duration": duration,
                "risk_score": f"{risk_score:.4f}"
            })

    @torch.no_grad()
    def evaluate(model, dataloader):
        model.eval()
        losses = []
        for batch in dataloader:
            with torch.autocast(
                device_type=DEVICE.type, dtype=DTYPE, enabled=(DEVICE.type == "cuda")
            ):
                out = model(**batch)
            loss_gathered = accelerator.gather(out.loss.unsqueeze(0))
            losses.extend(loss_gathered.detach().float().cpu().tolist())
        mean_loss = sum(losses) / max(len(losses), 1)
        ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
        return mean_loss, ppl

    epoch_times = []
    checkpoint_times = []
    checkpoint_path = os.path.join(checkpoint_dir, REMOTE_CHECKPOINT_FILENAME)

    upload_client = None
    checkpoint_writer = None
    if run_type == "spot":
        if CHECKPOINT_MODE == "async":
            upload_client = AsyncCheckpointClient(queue_size=ASYNC_UPLOAD_QUEUE_SIZE)
            checkpoint_writer = AsyncCheckpointWriter(
                checkpoint_path=checkpoint_path,
                upload_staging_dir=upload_staging_dir,
                checkpoint_times=checkpoint_times,
                record_timing_fn=record_timing,
                upload_client=upload_client,
                queue_size=ASYNC_CHECKPOINT_QUEUE_SIZE,
                remote_name=REMOTE_CHECKPOINT_FILENAME
            )
        elif CHECKPOINT_MODE == "fixed":
            checkpoint_writer = FixedIntervalCheckpointWriter(
                checkpoint_path=checkpoint_path,
                checkpoint_times=checkpoint_times,
                record_timing_fn=record_timing,
                remote_name=REMOTE_CHECKPOINT_FILENAME
            )
        elif CHECKPOINT_MODE == "adaptive":
            checkpoint_writer = KaplanMeierCheckpointWriter(
                checkpoint_path=checkpoint_path,
                checkpoint_times=checkpoint_times,
                record_timing_fn=record_timing,
                remote_name=REMOTE_CHECKPOINT_FILENAME,
                data_source="aws",
                risk_threshold=0.05,
                window_size=600,
                max_sample_time=MAX_SAMPLE_TIME
            )

    def build_checkpoint_payload(epoch_idx, step_idx, phase):
        return {
            "epoch": epoch_idx,
            "step": step_idx,
            "phase": phase,
            "model_state_dict": _to_cpu_state_dict(accelerator.unwrap_model(model).state_dict()),
            "optimizer_state_dict": _to_cpu_state_dict(optimizer.state_dict()),
            "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
        }



    def maybe_restore_remote_checkpoint():
        if os.path.exists(checkpoint_path):
            return True
        if run_type != "spot":
            return False
        if accelerator.is_main_process:
            aprint(f"No local spot checkpoint found for {model_name}; trying remote server...")
        return download_checkpoint_file(REMOTE_CHECKPOINT_FILENAME, checkpoint_path)

    def load_spot_checkpoint_if_present():
        start_epoch = 1
        start_step = 0
        current_phase = "fp"

        if run_type == "spot" and maybe_restore_remote_checkpoint() and os.path.exists(checkpoint_path):
            aprint(f"Found spot checkpoint at {checkpoint_path} for {model_name}. Resuming...")
            chkpt = torch.load(checkpoint_path, map_location="cpu")
            start_epoch = chkpt["epoch"]
            start_step = chkpt.get("step", 0)
            current_phase = chkpt["phase"]

            if current_phase == "qat":
                aprint(
                    "Resuming directly into QAT phase, setting up fake quantize modules BEFORE loading state dict..."
                )
                base_config = Int8DynamicActivationIntxWeightConfig(
                    weight_dtype=torch.int4,
                    weight_granularity=PerGroup(32),
                )
                quantize_(accelerator.unwrap_model(model), QATConfig(base_config, step="prepare"))

            accelerator.unwrap_model(model).load_state_dict(chkpt["model_state_dict"])
            # optimizer.load_state_dict(chkpt["optimizer_state_dict"]) # removed for simplicity with accelerate handling
            # scheduler.load_state_dict(chkpt["scheduler_state_dict"])
        else:
            if run_type == "spot":
                aprint("No spot checkpoint found. Starting from scratch.")

        return start_epoch, start_step, current_phase

    start_epoch, start_step, current_phase = load_spot_checkpoint_if_present()

    def save_spot_checkpoint(epoch_idx, step_idx, phase):
        if checkpoint_writer is not None:
            chkpt = build_checkpoint_payload(epoch_idx, step_idx, phase)
            checkpoint_writer.save_checkpoint(chkpt, epoch_idx, step_idx, phase)

    def train_one_epoch(
        model, dataloader, optimizer, scheduler, epoch_idx, phase_name, start_step=0
    ):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        last_save_time = time.time()
        total_batches = len(dataloader)

        with tqdm(
            total=total_batches,
            initial=start_step,
            desc=f"[{phase_name}] Epoch {epoch_idx}",
            disable=not accelerator.is_main_process
        ) as pbar:
            for step, batch in enumerate(dataloader, start=1):
                if step <= start_step:
                    continue

                step_t0 = time.time()

                with accelerator.accumulate(model):
                    with torch.autocast(
                        device_type=DEVICE.type, dtype=DTYPE, enabled=(DEVICE.type == "cuda")
                    ):
                        out = model(**batch)
                        loss = out.loss

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.is_main_process:
                    running += loss.detach().item()

                    step_dt = time.time() - step_t0
                    record_timing(phase_name, epoch_idx, step, "train_step", step_dt)

                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{running / (step - start_step):.4f}"})

                    if run_type == "spot":
                        checkpoint_triggered = False
                        current_risk = 0.0  # Initialize with a default value

                        if hasattr(checkpoint_writer, 'should_save'):
                            elapsed_since_save = time.time() - last_save_time
                            total_elapsed = time.time() - global_start_time
                            
                            # Capture the tuple (bool, float) from the KM writer
                            res = checkpoint_writer.should_save(elapsed_since_save, total_elapsed) [cite: 130]
                            
                            if isinstance(res, tuple):
                                checkpoint_triggered, current_risk = res
                            else:
                                checkpoint_triggered = res # Fallback for non-adaptive writers
                        else:
                            # Default fixed interval check [cite: 131]
                            checkpoint_triggered = (SAVE_EVERY_N_STEPS > 0 and step % SAVE_EVERY_N_STEPS == 0)

                        if checkpoint_triggered:
                            # Now current_risk is properly initialized and populated
                            record_timing(phase_name, epoch_idx, step, "checkpoint_start", 0.0, risk_score=current_risk) 
                            
                            save_spot_checkpoint(epoch_idx, step, phase_name)
                            
                            record_timing(phase_name, epoch_idx, step, "checkpoint_end", 0.0) 
                            last_save_time = time.time()

        return running / max(total_batches - start_step, 1)

    # -----------------------------
    # Phase 1: normal fine-tuning
    # -----------------------------
    global_start_time = time.time()

    if current_phase == "fp":
        aprint(f"Starting standard fine-tuning for {model_name}...")
        for epoch in range(start_epoch, NUM_EPOCHS_FP + 1):
            epoch_start_time = time.time()

            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, epoch, "fp", start_step
            )
            start_step = 0

            eval_t0 = time.time()
            val_loss, val_ppl = evaluate(model, eval_loader)
            eval_dt = time.time() - eval_t0
            record_timing("fp", epoch, 0, "evaluate", eval_dt)

            if run_type == "spot":
                save_spot_checkpoint(epoch + 1, 0, "fp")

            total_epoch_time = time.time() - epoch_start_time
            epoch_times.append(total_epoch_time)
            if accelerator.is_main_process:
                aprint(
                    f"[fp] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} time={total_epoch_time:.2f}s"
                )

        start_epoch = 1
        current_phase = "qat"

    # -----------------------------
    # Phase 2: QAT prepare -> train
    # -----------------------------
    if current_phase == "qat":
        if start_epoch == 1:
            aprint(f"Preparing model {model_name} for QAT...")
            base_config = Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                weight_granularity=PerGroup(32),
            )
            quantize_(model, QATConfig(base_config, step="prepare"))

        for epoch in range(start_epoch, NUM_EPOCHS_QAT + 1):
            epoch_start_time = time.time()

            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, epoch, "qat", start_step
            )
            start_step = 0

            eval_t0 = time.time()
            val_loss, val_ppl = evaluate(model, eval_loader)
            eval_dt = time.time() - eval_t0
            record_timing("qat", epoch, 0, "evaluate", eval_dt)

            if run_type == "spot":
                save_spot_checkpoint(epoch + 1, 0, "qat")

            total_epoch_time = time.time() - epoch_start_time
            epoch_times.append(total_epoch_time)
            if accelerator.is_main_process:
                aprint(
                    f"[qat] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} time={total_epoch_time:.2f}s"
                )

    if checkpoint_writer is not None:
        checkpoint_writer.flush()
    if upload_client is not None:
        upload_client.flush()

    # -----------------------------
    # Convert after QAT
    # -----------------------------
    aprint(f"Converting QAT model {model_name}...")
    convert_t0 = time.time()

    base_config = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=PerGroup(32),
    )
    quantize_(model, QATConfig(base_config, step="convert"))

    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    convert_time = time.time() - convert_t0
    total_time = time.time() - global_start_time

    if accelerator.is_main_process:
        aprint(f"Saved quantized {model_name} model to: {output_dir}")
        aprint(f"Finished processing {model_name} ({run_type}) in {total_time:.2f}s!\n")

    if checkpoint_writer is not None:
        checkpoint_writer.close()
    if upload_client is not None:
        upload_client.close()

    shutil.rmtree(upload_staging_dir, ignore_errors=True)

    del model
    del optimizer
    del scheduler
    del train_loader
    del eval_loader
    del raw
    del lm_ds
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "run_type": run_type,
        "total_time": total_time,
        "epoch_times": epoch_times,
        "checkpoint_times": checkpoint_times,
        "convert_time": convert_time,
    }


def print_statistics(results):
    table = PrettyTable()
    table.field_names = [
        "Model",
        "Type",
        "Total (s)",
        "Epoch Avg (s)",
        "Epoch Std (s)",
        "Chkpt Avg (s)",
        "Chkpt Std (s)",
        "Convert (s)",
    ]

    for r in results:
        t_tot = f"{r['total_time']:.2f}"

        ep_times = r["epoch_times"]
        if len(ep_times) > 0:
            ep_avg = f"{statistics.mean(ep_times):.2f}"
            ep_std = (
                f"{statistics.stdev(ep_times):.2f}" if len(ep_times) > 1 else "0.00"
            )
        else:
            ep_avg = ep_std = "N/A"

        cp_times = r["checkpoint_times"]
        if len(cp_times) > 0:
            cp_avg = f"{statistics.mean(cp_times):.2f}"
            cp_std = (
                f"{statistics.stdev(cp_times):.2f}" if len(cp_times) > 1 else "0.00"
            )
        else:
            cp_avg = cp_std = "N/A"

        t_conv = f"{r['convert_time']:.2f}"

        table.add_row(
            [
                r["model"].split("/")[-1],
                r["run_type"],
                t_tot,
                ep_avg,
                ep_std,
                cp_avg,
                cp_std,
                t_conv,
            ]
        )

    print("\n" + str(table) + "\n")


def main():
    global CHECKPOINT_MODE
    parser = argparse.ArgumentParser(description="Train and QAT pipeline")
    parser.add_argument(
        "--checkpointing",
        type=str,
        choices=["fixed", "async", "adaptive", "none"],
        default="none",
        help="The checkpointing method to use. If 'none', runs the baseline."
    )
    parser.add_argument(
        "--sim_id",
        type=str,
        default="default_sim",
        help="Unique ID for the simulation to avoid checkpoint conflicts"
    )
    parser.add_argument("--num_epochs_fp", type=int, default=3, help="Number of full precision epochs")
    parser.add_argument("--num_epochs_qat", type=int, default=3, help="Number of QAT epochs")
    parser.add_argument("--max_sample_time", type=float, default=float('inf'), help="Max simulation sample time")
    args = parser.parse_args()

    global CHECKPOINT_MODE
    CHECKPOINT_MODE = args.checkpointing
    
    global REMOTE_CHECKPOINT_FILENAME
    REMOTE_CHECKPOINT_FILENAME = f"latest_spot_checkpoint_{args.sim_id}.pt"

    global NUM_EPOCHS_FP, NUM_EPOCHS_QAT, MAX_SAMPLE_TIME
    NUM_EPOCHS_FP = args.num_epochs_fp
    NUM_EPOCHS_QAT = args.num_epochs_qat
    MAX_SAMPLE_TIME = args.max_sample_time
    
    global BASE_OUTPUT_DIR, BASE_CHECKPOINT_DIR
    BASE_OUTPUT_DIR = f"./qat_experiment_out/{args.sim_id}"
    BASE_CHECKPOINT_DIR = f"./spot_checkpoints/{args.sim_id}"
    
    run_type = "baseline" if args.checkpointing == "none" else "spot"

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)

    results = []

    for model_name in MODELS:
        res = run_training_pipeline(model_name, run_type)
        results.append(res)

    print_statistics(results)


if __name__ == "__main__":
    main()
