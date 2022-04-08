from torch import nn
from transformers import BertModel
import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CasrelCosine(nn.Module):
    def __init__(self, config):
        super(CasrelCosine, self).__init__()
        self.config = config
        self.bert_dim = 768
        with open(f"./data/{self.config.dataset}/relation_vectors", 'rb') as f:
          relation_vectors = np.load(f)
        self.encoded_relations = torch.Tensor(relation_vectors)
        self.bert_encoder = BertModel.from_pretrained("bert-base-uncased")
        
        self.sub_heads_linear = nn.Linear(self.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.bert_dim, 1)

        self.obj_heads_linear = nn.Linear(self.bert_dim + self.config.rel_num, self.config.rel_num)
        self.obj_tails_linear = nn.Linear(self.bert_dim + self.config.rel_num, self.config.rel_num) 

    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text):
        sub_head = torch.matmul(sub_head_mapping, encoded_text) # [batch_size, 1, bert_dim]
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text) # [batch_size, 1, bert_dim]
        sub = (sub_head + sub_tail) / 2 # [batch_size, 1, bert_dim]
        encoded_text = encoded_text + sub # [batch_size, seq_len, bert_dim]
        
        scores = self.scoring(encoded_text) # [batch_size, max_length, rel_num]
        
        pred_obj_heads = self.obj_heads_linear(scores)  # [batch_size, seq_len, rel_num]
        pred_obj_heads = torch.sigmoid(pred_obj_heads)
        pred_obj_tails = self.obj_tails_linear(scores)  # [batch_size, seq_len, rel_num]
        pred_obj_tails = torch.sigmoid(pred_obj_tails)
        return pred_obj_heads, pred_obj_tails

    def get_encoded_text(self, token_ids, mask):
        encoder = self.bert_encoder(token_ids, attention_mask=mask) # [batch_size, seq_len, bert_dim(768)]
        return encoder.last_hidden_state, encoder.pooler_output

    def get_subs(self, encoded_text):
        pred_sub_heads = self.sub_heads_linear(encoded_text)  # [batch_size, seq_len, 1]
        pred_sub_heads = torch.sigmoid(pred_sub_heads)
        pred_sub_tails = self.sub_tails_linear(encoded_text)  # [batch_size, seq_len, 1]
        pred_sub_tails = torch.sigmoid(pred_sub_tails)
        return pred_sub_heads, pred_sub_tails

    def scoring(self, encoded_text):
      encoded_relations_T = torch.transpose(self.encoded_relations, 0, 1) # [1, bert_dim, num_rels]
      rel_features = encoded_text.matmul(encoded_relations_T.to('cuda:0')) # [1, bert_dim, num_rels]
      rel_scores = torch.softmax(rel_features, 1)
      rel_scores = torch.cat((encoded_text, rel_scores), -1)
      return rel_scores

    def forward(self, data):
        # print("encoding")
        token_ids = data['token_ids'] # [batch_size, seq_len]
        mask = data['mask'] # [batch_size, seq_len]
        seq_output, pooled_output = self.get_encoded_text(token_ids, mask) # [batch_size, seq_len, bert_dim(768)]
        # ===================== Subject Tagger =====================
        # print("subject tagger")
        pred_sub_heads, pred_sub_tails = self.get_subs(seq_output)  # [batch_size, seq_len, 1]
        sub_head_mapping = data['sub_head'].unsqueeze(1)  # [batch_size, 1, seq_len]
        sub_tail_mapping = data['sub_tail'].unsqueeze(1)  # [batch_size, 1, seq_len]
        # ===================== Relation-specific Tagger =====================
        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, seq_output) # [batch_size, seq_len, rel_num]
        return pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails
