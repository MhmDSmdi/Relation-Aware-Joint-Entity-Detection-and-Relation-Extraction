# <p align=center>Relation Aware Joint Entity Detection and Relation Extraction </p>
# Demo
For replicating the reported results, you can simply open ``Run.ipynb`` in google colab, and follow its instruction to replicate the results.
# Use the trained model
1. Download the necessary files like pretrained models
```
# save pretrained models to checkpoints/
# models will take about 1GB
bash ./download_checkpoints.sh
```
2- Now, you can evaluate the model using the following bash command:
```
python ./test.py \
--model_name="Casrel_Rethinking" \
--rel_num=171 \
--path="./checkpoints/model_rethinking" \
--test_prefix="test_triples" \
--dataset="WebNLG" \
```
Sevaral important options includes:
* `--model_name`: the model that is used, CASREL or Rethinking
* `--path`: path of the downloaded checkpoint
* `--test_prefix`: the prefix of data file that you want to evaluate the model (you can use any file name in data/WebNLG without .json extension)
* `--dataset`: the dataset that you are using

# Acknowledge
In this project, we used the following sources:
* [PyTorch implementation](https://github.com/longlongman/CasRel-pytorch-reimplement) of the [original paper](https://arxiv.org/abs/1909.03227).
* Datasets and data processors provided by the [original CASREL](https://github.com/weizhepei/CasRel).