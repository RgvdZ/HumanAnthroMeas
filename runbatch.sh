#!/bin/bash
#SBATCH --job-name=gpu_execution      # Assigns a name to your job [cite: 246]
#SBATCH --partition=v100gpu           # Target partition: use v100gpu or p100gpu [cite: 246, 250]
#SBATCH --nodes=1                     # Run all processes on a single node [cite: 246]
#SBATCH --ntasks=1                    # Run a single task [cite: 246]
#SBATCH --cpus-per-task=4             # Adjust the number of CPU cores needed [cite: 246]
#SBATCH --gres=gpu                    # CRITICAL: This requests and allocates the GPU [cite: 246, 254]
#SBATCH --mem=32gb                    # Request appropriate RAM limit [cite: 246]
#SBATCH --time=2-01:00:00             # Walltime limit in day-hrs:min:sec (2 days 1 hour) [cite: 239, 268]
#SBATCH --output=gpu_job_%j.log       # Standard output and error log filename [cite: 246]

# 1. Activate your Python environment cleanly
source .venv/bin/activate

# 2. Move to your project repository directory
cd measurement/dataset/codebase/BMNet

# 3. Dynamically assign a random port between 10000 and 65000 to prevent collisions

# 5. Execute your main network training program
python3 main.py
