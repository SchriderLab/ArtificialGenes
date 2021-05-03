#!/bin/bash

# Launches slurm job to train conditional GAN
# Example: slurm/run_cgan.sh populations populations_output 100

#SBATCH --job-name=CGAN_AG_training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --partition=volta-gpu
#SBATCH --output=CGAN-%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end
#SBATCH --mail-user=nickmatt@live.unc.edu

unset OMP_NUM_THREADS

IDIR=$1
ODIR=$2

if [[ "$3" != "" ]]; then
    SAVE_FREQ="$3"
else
    SAVE_FREQ="0"
fi


# Set SIMG path
SIMG_PATH=/proj/dschridelab/SparseNets/pytorch1.4.0-py3-cuda10.1-ubuntu16.04_production.simg

echo singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 src/conditional_gan.py --idir $IDIR --odir $ODIR --save_freq $SAVE_FREQ --verbose --use_cuda --plot
singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 src/conditional_gan.py --idir $IDIR --odir $ODIR --save_freq $SAVE_FREQ --verbose --use_cuda --plot

