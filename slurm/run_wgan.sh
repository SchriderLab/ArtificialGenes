#!/bin/bash


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

ODIR=$1
IFILE=$2
SAVE_FREQ=$3

# Set SIMG path
SIMG_PATH=/proj/dschridelab/SparseNets/pytorch1.4.0-py3-cuda10.1-ubuntu16.04_production.simg

if [[ "$4" == "" ]]; then
    echo singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 src/train_wgan-gp.py --odir $ODIR --ifile $IFILE --save_freq $SAVE_FREQ --plot --verbose --use_cuda
    singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 src/train_wgan-gp.py --odir $ODIR --ifile $IFILE --save_freq $SAVE_FREQ --plot --verbose --use_cuda
else
    echo singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 src/train_wgan-gp.py --odir $ODIR --ifile $IFILE --save_freq $SAVE_FREQ --ag_size $4 --plot --verbose --use_cuda
    singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 src/train_wgan-gp.py --odir $ODIR --ifile $IFILE --save_freq $SAVE_FREQ --ag_size $4 --plot --verbose --use_cuda
fi
