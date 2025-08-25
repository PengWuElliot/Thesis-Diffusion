#!/bin/bash
#SBATCH --job-name=simplenet_mvtec
#SBATCH --output=slurm-%j.out              # Output file (%j = job ID)
#SBATCH --error=slurm-%j.err               # Error file
#SBATCH --time=20:00:00                    # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH --mem=50G                          # Memory limit

srun python inference.py
