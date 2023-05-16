#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch
#SBATCH --account=ece_gy_9143-2023sp
#SBATCH --partition=n2c48m24

singularity exec --nv \
	    --overlay /scratch/xd2152/pytorch-example/my_pytorch.ext3:ro \
	    /share/apps/images/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh;conda activate myenv;\
           python tools/test.py \
                --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
                TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth \
                TEST.USE_GT_BBOX False;"
