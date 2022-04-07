from torch import nn
from transformers import BertModel
import torch
import numpy as np
import os


class CasrelRethinking(nn.Module):
    def __init__(self, config):
        super(CasrelRethinking, self).__init__()
        self.config = config
        self.bert_dim = 768
        self.bert_encoder = BertModel.from_pretrained("bert-base-cased")
        
        self.rel_linear = nn.Linear(self.bert_dim, self.config.rel_num)
        with open(f"./data/{self.config.dataset}/relation_vectors", 'rb') as f:
          relation_vectors = np.load(f)
        self.rel_embedding = nn.Parameter(torch.from_numpy(relation_vectors).to(device='cuda:0'))

        self.sub_heads_linear = nn.Linear(self.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.bert_dim, 1)
        
        # Phase 1
        self.obj_heads_linear1 = nn.Linear(self.bert_dim, self.config.rel_num)
        self.obj_tails_linear1 = nn.Linear(self.bert_dim, self.config.rel_num)

        # Phase 2
        self.obj_heads_linear2 = nn.Linear(self.bert_dim +  self.config.rel_num, self.config.rel_num)
        self.obj_tails_linear2 = nn.Linear(self.bert_dim + self.config.rel_num, self.config.rel_num)
        
        
    def get_potential_relations(self, pooled_output):
        rel_logits = self.rel_linear(pooled_output)  # [batch_size, rel_num, 1]
        pred_rels = torch.sigmoid(rel_logits)  # [batch_size, rel_num, 1]
        return pred_rels

    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text):
        sub_head = torch.matmul(sub_head_mapping, encoded_text) # [batch_size, 1, bert_dim]
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text) # [batch_size, 1, bert_dim]
        sub = (sub_head + sub_tail) / 2 # [batch_size, 1, bert_dim]
        encoded_text = encoded_text + sub # [batch_size, seq_len, bert_dim]

        # ===================== Phase1: Head Object Prediction =====================
        pred_obj_heads1 = self.obj_heads_linear1(encoded_text)  # [batch_size, seq_len, rel_num]
        pred_obj_heads1 = torch.sigmoid(pred_obj_heads1)

        # ===================== Phase1: Tail Object Prediction =====================
        pred_obj_tails1 = self.obj_tails_linear1(encoded_text)  # [batch_size, seq_len, rel_num]
        pred_obj_tails1 = torch.sigmoid(pred_obj_tails1)

        # ===================== Phase2: Feature Enhancing =====================
        # sub = sub.expand(-1, encoded_text.size()[1], self.bert_dim)  # [batch_size, seq_len, bert_dim]
        pred_obj_heads_features = torch.cat((encoded_text, pred_obj_heads1.clone()), dim=2)  # [batch_size, seq_len, bert_dim + rel_num]
        pred_obj_tails_features = torch.cat((encoded_text, pred_obj_tails1.clone()), dim=2)  # [batch_size, seq_len, bert_dim + rel_num]

        # ===================== Phase2: Head Object Prediction =====================
        pred_obj_heads2 = self.obj_heads_linear2(pred_obj_heads_features)  # [batch_size, seq_len, rel_num]
        pred_obj_heads2 = torch.sigmoid(pred_obj_heads2)

        # ===================== Phase2: Tail Object Prediction =====================
        pred_obj_tails2 = self.obj_tails_linear2(pred_obj_tails_features)  # [batch_size, seq_len, rel_num]
        pred_obj_tails2 = torch.sigmoid(pred_obj_tails2)
        return pred_obj_heads1, pred_obj_tails1, pred_obj_heads2, pred_obj_tails2

    def get_subs(self, encoded_text):
        pred_sub_heads = self.sub_heads_linear(encoded_text)  # [batch_size, seq_len, 1]
        pred_sub_heads = torch.sigmoid(pred_sub_heads)

        pred_sub_tails = self.sub_tails_linear(encoded_text)  # [batch_size, seq_len, 1]
        pred_sub_tails = torch.sigmoid(pred_sub_tails)
        return pred_sub_heads, pred_sub_tails

    def get_encoded_text(self, token_ids, mask):
        encoder = self.bert_encoder(token_ids, attention_mask=mask) # [batch_size, seq_len, bert_dim(768)]
        return encoder.last_hidden_state, encoder.pooler_output

    def get_rel_prediction(self, pooled_output, seq_output, ground_truth_rels=None):
        pred_rels = self.get_potential_relations(pooled_output)
        if ground_truth_rels != None:
          rel_embs = ground_truth_rels @ self.rel_embedding # [batch_size, bert_dim]
        else:
          rel_embs = pred_rels @ self.rel_embedding # [batch_size, bert_dim]
        rel_embs = rel_embs.unsqueeze(1).expand(-1, seq_output.size()[1], self.bert_dim)
        seq_output = seq_output + rel_embs
        # ===================== Adding Context =====================
        # pooled_output = pooled_output.unsqueeze(1).expand(-1, seq_output.size()[1], self.bert_dim)
        # decode_input = torch.cat((decode_input, pooled_output), dim=2)
        # ==========================================================
        return pred_rels, seq_output

    def forward(self, data):
        token_ids = data['token_ids'] # [batch_size, seq_len]
        mask = data['mask'] # [batch_size, seq_len]
        seq_output, pooled_output = self.get_encoded_text(token_ids, mask)
        # ===================== Subject Tagger =====================
        pred_sub_heads, pred_sub_tails = self.get_subs(seq_output)  # [batch_size, seq_len, 1]
        sub_head_mapping = data['sub_head'].unsqueeze(1)  # [batch_size, 1, seq_len]
        sub_tail_mapping = data['sub_tail'].unsqueeze(1)  # [batch_size, 1, seq_len]
        # ===================== Relation Prediction =====================
        pred_rels, seq_output = self.get_rel_prediction(pooled_output, seq_output, data['pot_rels'])
        # ===================== Relation-specific Tagger =====================
        pred_obj_heads1, pred_obj_tails1, pred_obj_heads2, pred_obj_tails2 = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, seq_output) # [batch_size, seq_len, rel_num]
        return pred_sub_heads, pred_sub_tails, pred_obj_heads1, pred_obj_tails1, pred_obj_heads2, pred_obj_tails2, pred_rels
