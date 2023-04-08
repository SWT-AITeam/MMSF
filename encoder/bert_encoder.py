import logging
import copy
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import json, os


class BERTEncoder(nn.Module):
    def __init__(self, pretrain_path, vec_path):

        super().__init__()
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.vecpath = vec_path

    def forward(self, token, att_mask):
        x = self.bert(token, attention_mask=att_mask)
        x = x[0]
        out = self.vec(x, token)
        return out

    def vec(self, vectors, tokens):
        tokens1 = []
        vectors1 = []
        with open(self.vecpath, 'r') as fout:
            dic = json.load(fout)
            for i in range(len(dic)):
                tokens1.append(dic[i]['token'])
                vectors1.append(dic[i]['vec'])
        mps = []
        for token in tokens:
            value2index = {}
            for idx, t in enumerate(token.cpu().numpy().tolist()):
                value2index[t] = idx
            mps.append(value2index)
        for mp, token, vec, token1, vec1 in zip(mps, tokens, vectors, tokens1, vectors1):
            for i, t in enumerate(token1):
                if t in mp:
                    idx = mp[t]
                    if i < len(vec1):
                        vec[idx] = vec1[i]
                    else:
                        vec[idx] = vec1[0]
        return vectors


