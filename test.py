import config
import framework
import argparse
import models
import os
import torch
import numpy as np
import random
from pprint import pprint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Casrel', help='name of the model')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='NYT')
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--test_epoch', type=int, default=5)
parser.add_argument('--train_prefix', type=str, default='train_triples')
parser.add_argument('--dev_prefix', type=str, default='dev_triples')
parser.add_argument('--test_prefix', type=str, default='test_triples')
parser.add_argument('--max_len', type=int, default=150)
parser.add_argument('--rel_num', type=int, default=44)
parser.add_argument('--period', type=int, default=50)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--path', type=str, default="")
args = parser.parse_args()

con = config.Config(args)
pprint(vars(con), open(con.hyperparameters_save_name, 'w'))
if torch.cuda.is_available():
    print("GPU is available.")
    fw = framework.Framework(con)
    model = {
        'Casrel': models.Casrel,
        'Casrel_Cosine': models.CasrelCosine,
        'Casrel_Potential': models.CasrelPotentialRelation,
        'Casrel_Rethinking': models.CasrelRethinking
    }
    fw.testall(model[args.model_name], con.saved_model)
else:
    print("GPU is not available.")
