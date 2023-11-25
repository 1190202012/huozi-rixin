#!/bin/bash
#SBATCH -J trial
#SBATCH -o test-%j.print
#SBATCH -e test-%j.terminal
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --mem=256GB
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a100-sxm4-80gb:2


#source ~/.bashrc

#conda activate rixin

python run.py -c config/demo.yaml