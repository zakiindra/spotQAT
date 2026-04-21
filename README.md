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

## Running Preemption Simulators

To assess how well your checkpointing method withstands preemptions across an endless simulation, you can run one of the preemption simulator scripts. These orchestrators launch `train_and_qat_modified.py` and gracefully recreate Spot termination patterns.

They accept two primary optional arguments:
- `--checkpointing-method`: Determines how models will create their resume states. (e.g. `fixed`, `async`, `adaptive`).
- `--max-sample-time`: Enforces a simulated instance ceiling. If the stochastic distribution selects a lifetime longer than this value (in seconds), it will resample.

### Usage Examples
```bash
# AWS Trace Simulator
python aws_preemption.py --checkpointing-method=adaptive --max-sample-time=7200

# Google Trace Simulator 
python google_preemption.py --checkpointing-method=async

# Poisson Distribution Simulator
python poisson_preemption.py --checkpointing-method=fixed

# Generalized Spot Trace Simulator
python spot_preemption.py --checkpointing-method=adaptive
```
