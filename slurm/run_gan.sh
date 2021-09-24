#!/bin/bash

# Launches slurm job to train vanilla GAN
# Example: slurm/run_gan.sh input_file gan_output 200

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
#SBATCH --mail-user=nickmatt@live.unc.edu

unset OMP_NUM_THREADS

IFILE=$1
ODIR=$2

if [[ "$3" != "" ]]; then
    SAVE_FREQ="$3"
else
    SAVE_FREQ="0"
fi


# Set SIMG path
SIMG_PATH=/proj/dschridelab/SparseNets/pytorch1.4.0-py3-cuda10.1-ubuntu16.04_production.simg

echo singularity exec --nv -B /pine -B /proj -B /overflow/dschridelab/users/wwbooker $SIMG_PATH python3 src/train_gan.py --odir $ODIR --ifile $IFILE --save_freq $SAVE_FREQ --plot --verbose --use_cuda
singularity exec --nv -B /pine -B /proj -B /overflow/dschridelab/users/wwbooker $SIMG_PATH python3 src/train_gan.py --odir $ODIR --ifile $IFILE --save_freq $SAVE_FREQ --plot --verbose --use_cuda
