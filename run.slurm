#!/bin/bash
#SBATCH --job-name=simplenet_mvtec
#SBATCH --output=slurm-%j.out              # Output file (%j = job ID)
#SBATCH --error=slurm-%j.err               # Error file
#SBATCH --time=20:00:00                    # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH --partition=gpu-a100-80g
#SBATCH --mem=50G                          # Memory limit
export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_DIR="/scratch/work/wup5/dataset/triton/broke/bad"
export MASK_DIR="/scratch/work/wup5/dataset/triton/broke/mask"
export OUTPUT_DIR="saved_model"

accelerate launch train_dreambooth_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --mask_data_dir=$MASK_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="surface break" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=20000\
  --checkpointing_step=1000000
