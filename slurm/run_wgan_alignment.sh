#!/bin/bash

# Launchs job to train Wasserstein GAN
# Example: slurm/run_wgan.sh input_data wgan_output 200 250

#SBATCH --job-name=AG_training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --partition=volta-gpu
#SBATCH --output=AG-%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end
#SBATCH --mail-user=wbooker14@gmail.com

unset OMP_NUM_THREADS

IDIR=$1
ODIR=$2
SAVE_FREQ=$3

# Set SIMG path
SIMG_PATH=/proj/dschridelab/SparseNets/pytorch1.4.0-py3-cuda10.1-ubuntu16.04_production.simg

if [[ "$4" == "" ]]; then
    echo singularity exec --nv -B /pine -B /proj -B /overflow $SIMG_PATH python3 src/train_wgan-gp-alignment.py --odir $ODIR --idir $IDIR --save_freq $SAVE_FREQ --plot --verbose --use_cuda
    singularity exec --nv -B /pine -B /proj -B /overflow $SIMG_PATH python3 src/train_wgan-gp-alignment.py --odir $ODIR --idir $IDIR --save_freq $SAVE_FREQ --plot --verbose --use_cuda
else
    echo singularity exec --nv -B /pine -B /proj -B /overflow $SIMG_PATH python3 src/train_wgan-gp-alignment.py --odir $ODIR --idir $IDIR --save_freq $SAVE_FREQ --ag_size $4 --plot --verbose --use_cuda
    singularity exec --nv -B /pine -B /proj -B /overflow $SIMG_PATH python3 src/train_wgan-gp-alignment.py --odir $ODIR --idir $IDIR --save_freq $SAVE_FREQ --ag_size $4 --plot --verbose --use_cuda
fi
