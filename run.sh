#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=23:30:0
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

module purge
module load python/3

# export PYTHONPATH="/cvmfs/soft.computecanada.ca/custom/python/site-packages:/home/mhmd/projects/def-drafiei/mhmd/relation-extraction/CasRel-Torch"
source ./env/bin/activate

python ./train.py \
--model_name="Casrel_Potential" \
--batch_size=6 \
--max_epoch=50 \
--test_epoch=10 \
--max_len=64 \
--rel_num=24 \
--dataset="NYT" \