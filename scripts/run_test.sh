#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=00:10:0
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

module purge
module load python/3


source ./env/bin/activate

python ./test.py \
--model_name="Casrel_Rethinking" \
--batch_size=6 \
--max_epoch=50 \
--test_epoch=10 \
--max_len=64 \
--rel_num=171 \
--path="./checkpoints/model_rethinking" \
--dataset="WebNLG" \
