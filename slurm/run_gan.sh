#!/bin/bash


#SBATCH --job-name=AG_training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --partition=volta-gpu
#SBATCH --output=run-%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end
#SBATCH --mail-user=nickmatt@live.unc.edu

unset OMP_NUM_THREADS

sbatch -p general -N 1 --mem 5120 -n 1 -t 2:00:00 --mail-type=end --mail-user=onyen@email.unc.edu --wrap="python3 ../src/gan_script.py"
