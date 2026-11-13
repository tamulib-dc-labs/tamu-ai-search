#!/bin/bash
# Embed every chunk with one model. The output, $DATA/vectors-<model>.npy, goes back next to Solr
# for `tamu-ai load`.
# Submit from this folder:  sbatch embed.slurm
# Another model:            sbatch --export=ALL,MODEL=BAAI/bge-m3 embed.slurm
#SBATCH --job-name=embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=embed.%j.log

source ./env.sh
source "$VENV/bin/activate"
module load WebProxy

tamu-ai embed "$DATA" --model "${MODEL:-BAAI/bge-base-en-v1.5}"
