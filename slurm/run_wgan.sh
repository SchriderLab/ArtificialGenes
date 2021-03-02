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

# if 2nd argument is 0, models are not saved
# if 3rd argument is given, we run wgan-gp, otherwise we run wgan-gc

unset OMP_NUM_THREADS

ODIR=$1
SAVE_FREQ=$2

# Set SIMG path
SIMG_PATH=/proj/dschridelab/SparseNets/pytorch1.4.0-py3-cuda10.1-ubuntu16.04_production.simg

if [[ "$3" != "" ]]; then
    echo singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 ../src/train_wgan-gp.py --odir $ODIR --save_freq $SAVE_FREQ --plot --verbose
    singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 ../src/train_wgan-gp.py --odir $ODIR --save_freq $SAVE_FREQ --plot --verbose
else
    echo singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 ../src/train_wgan.py --odir $ODIR --save_freq $SAVE_FREQ --plot --verbose
    singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 ../src/train_wgan.py --odir $ODIR --save_freq $SAVE_FREQ --plot --verbose
fi
