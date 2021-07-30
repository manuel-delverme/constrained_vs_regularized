ToDO: add description

# Quick Start

```shell
# Setup env from environment.yml
conda env create --prefix ./env --file environment.yml
conda activate ./env

# Alternatively, install requirements
# pip install --upgrade -r requirements.txt

# Setup WandB + select desired project
wandb init

# Run for a specific config_fairness.py
python main_fairness.py

# Sweep with sweep.yml
wandb sweep sweep.yml
export DISPLAY=""
export BUDDY_CURRENT_TESTING_BRANCH="local_sweep"
wandb agent id # id is printed in the previous step.
```