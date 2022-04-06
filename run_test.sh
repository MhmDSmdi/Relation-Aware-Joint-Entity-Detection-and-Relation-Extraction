#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=00:10:0
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

module purge
module load python/3

# export PYTHONPATH="/cvmfs/soft.computecanada.ca/custom/python/site-packages:/home/mhmd/projects/def-drafiei/mhmd/relation-extraction/CasRel-Torch"
source ./env/bin/activate

python ./test.py \
--model_name="Casrel_Rethinking" \
--batch_size=6 \
--max_epoch=50 \
--test_epoch=10 \
--max_len=64 \
--rel_num=171 \
--path="/home/mhmd/projects/def-drafiei/mhmd/relation-extraction/CasRel-Torch/final_experiments/WebNLG-Casrel_Rethinking-6/20220402130856/model_WebNLG" \
--dataset="WebNLG" \