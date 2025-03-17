#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1


srun --export=ALL python3 main_darai.py
