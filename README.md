# <p align=center>Relation Aware Joint Entity Detection and Relation Extraction </p>

<p align="center"><img width="70%" src="https://raw.githubusercontent.com/MhmDSmdi/Relation-Aware-Joint-Entity-Detection-and-Relation-Extraction/main/figures/overview.png" /></p>

# Demo
For replicating the reported results, you can simply open ``Run.ipynb`` in Google Colab, and follow its instruction to replicate the results.
# Use the trained model
1. Download the necessary files like pre-trained models
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
Several important options include:
* `--model_name`: the model that is used, CASREL or Rethinking
* `--path`: the path of the downloaded checkpoint
* `--test_prefix`: the prefix of the data file that you want to evaluate the model (you can use any file name in data/WebNLG without the ``.json`` extension)
* `--dataset`: the dataset that you are using

The expected results would be as follow:
```
correct_num: 1406, predict_num: 1572, gold_num: 1581
f1: 0.8918, precision: 0.8944, recall: 0.8893
```

# Train models from scratch
You can train a model from scratch using the following bash command:
```
python ./train.py \
--model_name="Casrel_Rethinking" \
--batch_size=6 \
--max_epoch=50 \
--test_epoch=10 \
--max_len=64 \
--rel_num=24 \
--dataset="NYT" \
```

# Acknowledge
In this project, we used the following sources:
* [PyTorch implementation](https://github.com/longlongman/CasRel-pytorch-reimplement) of the [original paper](https://arxiv.org/abs/1909.03227).
* Datasets and data processors provided by the [original CASREL](https://github.com/weizhepei/CasRel).