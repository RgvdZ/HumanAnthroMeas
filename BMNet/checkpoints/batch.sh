#!/bin/bash
#SBATCH --job-name=convert_model       # Name of the job
#SBATCH --partition=p100gpu            # Use P100 GPU partition
#SBATCH --nodes=1                      # Single node
#SBATCH --ntasks=1                     # Single task
#SBATCH --cpus-per-task=4              # 4 CPU cores
#SBATCH --gres=gpu:1                   # Request 1 GPU (explicit count)
#SBATCH --mem=16gb                     # 16GB RAM (enough for conversion)
#SBATCH --time=00:30:00                # Max 15 minutes (conversion is fast)
#SBATCH --output=convert_%j.log        # Log file with job ID

# Change to the directory where this script and the model files are located

cd /iitgn/homedirs/preyum.kumar/rgvd

# Activate your Python environment
source .venv/bin/activate   # <--- REPLACE with your actual venv path
# Or if you use conda:

cd /iitgn/homedirs/preyum.kumar/rgvd/measurement/dataset/codebase/BMNet/checkpoints

# Run the conversion script
python3 mobile.py

