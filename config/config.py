from datetime import datetime
import os

class Config(object):
    def __init__(self, args):
        self.args = args
        
        # train hyper parameter
        self.multi_gpu = args.multi_gpu
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.rel_num = args.rel_num
        self.model_name = args.model_name
        self.saved_model = args.path
        #self.abl_emb = args.abl_emb
        #self.abl_rethink = args.abl_rethink
        
        # dataset
        self.dataset = args.dataset

        # path and name
        self.root = './'
        self.data_path = self.root + '/data/' + self.dataset

        self.train_prefix = "train_triples"
        self.dev_prefix = "dev_triples"
        self.test_prefix = args.test_prefix

        self.experiment_path = f"{self.root}/experiments/{self.dataset}-{self.model_name}-{self.batch_size}"

        if not os.path.exists(self.experiment_path):
          os.makedirs(self.experiment_path)
        self.checkpoint_dir = self.experiment_path
        self.log_dir = self.experiment_path
        self.result_dir = self.experiment_path
        self.model_save_name = f"model_{self.dataset}"
        self.log_save_name = f"log_{self.dataset}"
        self.result_save_name = f"output_{self.dataset}.json"
        self.hyperparameters_save_name = f"{self.experiment_path}/params_{self.dataset}.out"
        
        # log setting
        self.period = args.period
        self.test_epoch = args.test_epoch

        # debug
        self.debug = False
        if self.debug:
            self.dev_prefix = self.train_prefix
            self.test_prefix = self.train_prefix

