#!/bin/bash
#SBATCH --job-name=try_cat
#SBATCH --partition=u22
#SBATCH --mem=1G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%j.out

ls /data3/dataset/the_pile_deduplicated/data
