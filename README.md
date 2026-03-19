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
