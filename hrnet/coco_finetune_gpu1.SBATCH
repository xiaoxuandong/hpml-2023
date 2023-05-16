#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu
#SBATCH --job-name=finetune
#SBATCH --account=ece_gy_9143-2023sp
#SBATCH --partition=n1s8-v100-1

singularity exec --nv \
            --overlay /scratch/xd2152/pytorch-example/my_pytorch.ext3:ro \
            /share/apps/images/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh;conda activate myenv;\
           python tools/train.py \
                --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3_finetune_gpu1.yaml \
OUTPUT_DIR output-gpu1-bs80\
  LOG_DIR log-gpu1-bs80\
  TRAIN.END_EPOCH 10 \
  TRAIN.BATCH_SIZE_PER_GPU 80"
