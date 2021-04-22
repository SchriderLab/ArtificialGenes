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

if [[ "$2" != "" ]]; then
    SAVE_FREQ="$2"
else
    SAVE_FREQ="0"
fi


# Set SIMG path
SIMG_PATH=/proj/dschridelab/SparseNets/pytorch1.4.0-py3-cuda10.1-ubuntu16.04_production.simg

if [[ "$3" != "" ]]; then
    echo singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 src/train_gan.py --odir $ODIR --save_freq $SAVE_FREQ --verbose --use_cuda
    singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 src/train_gan.py --odir $ODIR --save_freq $SAVE_FREQ --verbose --use_cuda
else
    echo singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 src/train_gan.py --odir $ODIR --save_freq $SAVE_FREQ --plot --verbose --use_cuda
    singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 src/train_gan.py --odir $ODIR --save_freq $SAVE_FREQ --plot --verbose --use_cuda
fi
