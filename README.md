# SpotQAT
A TorchAO Quantization-Aware Training experiment.

## Setup

This project uses `uv` for dependency management. To synchronize dependencies and set up the virtual environment, run:

```bash
uv sync
```

This will create a virtual environment in `.venv` and install all required packages.

To run scripts using the created environment, use the python executable inside the `.venv`:

```bash
uv run train_and_qat.py
```

To install new packages, use:

```bash
uv pip install <package_name>
```

## Running the Pipeline

You can run the modified training script natively using standard Python. It will automatically leverage Accelerate for Multi-GPU handling.

### Baseline (No Checkpoints)
To test the pipeline without generating internal checkpoints (baseline mode):
```bash
python train_and_qat_modified.py --checkpointing=none
```

### Spot Checkpointing Mode
To run the framework saving Spot instance checkpoints periodically using a specified method:
```bash
python train_and_qat_modified.py --checkpointing=fixed
# Options: fixed, async, adaptive, none
```

For advanced usage, training lengths and configurations can be tweaked from the command line:
```bash
python train_and_qat_modified.py --checkpointing=adaptive --num_epochs_fp=3 --num_epochs_qat=3 --sim_id=run1
```

## Running Preemption Simulators

To assess how well your checkpointing method withstands preemptions across an endless simulation, you can run one of the preemption simulator scripts. These orchestrators launch `train_and_qat_modified.py` and gracefully recreate Spot termination patterns.

They accept the following arguments:
- `--checkpointing-method`: Determines how models will create their resume states. (e.g. `fixed`, `async`, `adaptive`).
- `--max-sample-time`: Enforces a simulated instance ceiling. If the stochastic distribution selects a lifetime longer than this value (in seconds), it will resample. The Kaplan-Meier adaptive checkpointer dynamically re-filters its survival curve whenever this constraint is adjusted!
- `--sim_id`: Uniquely footprints the remote checkpoints generated. Defends parallel executions from overwriting each other.
- `--gpu_id`: Forces `CUDA_VISIBLE_DEVICES` exclusively onto the simulated node allowing seamless standalone multitenancy across your system.
- `--num_epochs_fp` & `--num_epochs_qat`: Specifies exactly how long the fine tuning and quantization training loops will run for each model!

### Usage Examples
```bash
# AWS Trace Simulator
python aws_preemption.py --checkpointing-method=adaptive --max-sample-time=7200 --gpu_id=0 --sim_id=aws_sim1

# Google Trace Simulator 
python google_preemption.py --checkpointing-method=async --gpu_id=1

# Poisson Distribution Simulator
python poisson_preemption.py --checkpointing-method=fixed --sim_id=poisson_baseline

# Generalized Spot Trace Simulator
python spot_preemption.py --checkpointing-method=adaptive --num_epochs_fp=5 --num_epochs_qat=10
```

### Full Execution Script
To run a complete end-to-end simulation easily with all the provided features plugged in, use the included shell wrapper!
```bash
./run_simulation.sh
```
