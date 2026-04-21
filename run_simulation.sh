#!/bin/bash
# run_simulation.sh
# Example script to run a preemption simulation with all configurable arguments

echo "Starting Google Preemption Simulator..."
python3 google_preemption.py \
    --checkpointing-method adaptive \
    --max-sample-time 3600 \
    --gpu_id 0 \
    --sim_id test_sim_001 \
    --num_epochs_fp 3 \
    --num_epochs_qat 3

echo "Simulation orchestration finished."
