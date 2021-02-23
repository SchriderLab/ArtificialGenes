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

# Set SIMG name
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/2.3.1/simg/tensorflow2.3.1-py3-cuda10.1-ubuntu18.04.simg

echo singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 ../src/gan_script.py 
singularity exec --nv -B /pine -B /proj $SIMG_PATH python3 ../src/gan_script.py 