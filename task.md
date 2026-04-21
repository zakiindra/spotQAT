TASK: [DONE] Read all python, bash, and md files in Emulator-unified folder. Only performs head to csv file to understand the data format. This is a folder that was given by someone as reference to implement preemption simulator based on AWS spot instance trace data. I want to implement the preemption simulator based on this that follows the structure and logic in google_preemption.py. Create the file as aws_preemption.py.

TASK: [DONE] Refactor train_and_qat_modified.py to separate the different existing checkpointing method into different class in different files.

TASK: [DONE] Implement checkpointing method that adapts to spot preemption by leveraging Kaplan-Meier survival analysis on preemption data. For now, use the data in data/gcp/data.json and csv data in Emulator-unified.

TASK: [DONE] Add script to setup environment using conda, for deployment in system where only conda is available and there is no sudo access. The script should install all the dependencies in a local directory.

TASK: [DONE] Rename setup_env.sh to setup_conda_env.sh and make it executable.

TASK: [DONE] Refactor train_and_qat_modified.py to only run baseline by default. Checkpointing should be declared as optional argument. If checkpointing is enabled, it should run the training with chosen checkpointing method. The training should only run single training based on chosen checkpointing method. Keep the list of models, we will refactor again later. Keep the logic to print report and statistics.

TASK: [DONE] Refactor train_and_qat_modified.py to support training with multiple GPUs. Use Hugging Face Accelerate for the multi-GPU training. For additional context, I have 3x RTX A6000 GPUs. The training should work when only 1 GPU is available.

TASK: update uv.lock file and setup_conda_env.sh file to include prettytable and accelerate library