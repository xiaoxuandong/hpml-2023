#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output="%x.txt"
#SBATCH --mem=16GB
#SBATCH --gres=gpu
#SBATCH --job-name=bs128
module purge
cd /scratch/yz8458/final/EfficientNet/src
singularity exec --nv \
            --overlay /scratch/yz8458/pytorch-example/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python train.py --cuda --pretrained --batch_size 128;"



#!/bin/bash

#SBATCH --output="%x.txt"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:v100:4
#SBATCH --account=ece_gy_9143-2023sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --job-name=hrNet

cd /home/yz8458/final/hrNet

singularity exec --nv \
            --overlay /scratch/yz8458/pytorch-example/my_pytorch.ext3:ro \
            /share/apps/images/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh;conda activate hrNet; \
            python tools/test.py \
                --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml "