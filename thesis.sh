#!/bin/bash
#SBATCH --job-name=exp5_ngc
#SBATCH --output=exp5_ngc.out
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bruno.jerkovic@ru.nl
#SBATCH --time=30:00:00
# your job goes here:
python3.8 main.py
