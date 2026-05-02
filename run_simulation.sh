#!/bin/bash
# run_simulation.sh
# Example script to run a preemption simulation with all configurable arguments

echo "Starting Google Preemption Simulator..."

python3 google_preemption.py \
    --checkpointing-method young_daly \
    --max-sample-time 3600 \
    --delta 44.5 \
    --mttf 1260.86 \
    --gpu_id 0 \
    --sim_id gcp_young_daly_async_001 \
    --num_epochs_fp 3 \
    --num_epochs_qat 3

sleep 60

python3 google_preemption.py \
    --checkpointing-method young_daly_async \
    --max-sample-time 3600 \
    --delta 44.5 \
    --mttf 1260.86 \
    --gpu_id 0 \
    --sim_id gcp_young_daly_async_001 \
    --num_epochs_fp 3 \
    --num_epochs_qat 3

sleep 60

python3 google_preemption.py \
    --checkpointing-method adaptive \
    --max-sample-time 3600 \
    --risk-threshold 0.02 \
    --scale-factor 24.0 \
    --min-interval 60 \
    --gpu_id 0 \
    --sim_id gcp_adaptive_sync_001 \
    --num_epochs_fp 3 \
    --num_epochs_qat 3

sleep 60

python3 google_preemption.py \
    --checkpointing-method adaptive_async \
    --max-sample-time 3600 \
    --risk-threshold 0.02 \
    --scale-factor 24.0 \
    --min-interval 60 \
    --gpu_id 0 \
    --sim_id gcp_adaptive_async_001 \
    --num_epochs_fp 3 \
    --num_epochs_qat 3

sleep 60

echo "Simulation orchestration finished."
