# -*- coding: utf-8 -*-
""" 
@Time    : 2022/5/15 21:48
@Author  : ZHAOXUAN
@FileName: data_pic.py
@SoftWare: PyCharm
"""
import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
from random import choice
from transformers import BertTokenizer
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from configs import Config
from framework.dataloaders import SentenceREDataset
BERT_MAX_LEN = 512

class PicDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """

    def __init__(self, path, rel2id_path, pretrain_path):
        super().__init__()
        self.path = path
        self.berttokenizer = BertTokenizer.from_pretrained(pretrain_path)#, do_lower_case=False, do_basic_tokenize = False)
        self.rel2id = json.load(open(rel2id_path))
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        # Load the file
        f = open(path, encoding='utf-8')
        print("打开")
        self.data = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()
        self.data_length = len(self.data)

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        item = self.data[index]
        seq = list(self.tokenizer(item))
        print("ss")
        return [self.rel2id[item['relation']]] + seq  # label, seq1, seq2, ...
    def tokenizer(self, item):
        # Sentence -> token
        sentence = item['token']
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        ent0 = ' '.join(sentence[pos_head[0]:pos_head[1]])
        ent1 = ' '.join(sentence[pos_tail[0]:pos_tail[1]])
        sent = ' '.join(sentence)
        word1 = ent0

        word2 = ent1
        print("sent", sent)
        re_tokens = self._tokenize(sent)

        if len(re_tokens) > BERT_MAX_LEN:
            re_tokens = re_tokens[:BERT_MAX_LEN]

        ent0 = self._tokenize(ent0)[1:-1]
        ento_tokens = self.berttokenizer.convert_tokens_to_ids(ent0)
        print('ento_tokens',ento_tokens)

        print('ent0', ent0)
        ent1 = self._tokenize(ent1)[1:-1]

        heads_s = self.find_head_idx(re_tokens, ent0)
        print('heads_s', heads_s)
        heads_e = heads_s + len(ent0) - 1

        tails_s = self.find_head_idx(re_tokens, ent1)
        tails_e = tails_s + len(ent1) - 1

        indexed_tokens = self.berttokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        heads = torch.zeros(avai_len).float()  # (B, L)
        tails = torch.zeros(avai_len).float()  # (B, L)

        for i in range(avai_len):
            if i >= heads_s and i <= heads_e:
                heads[i] = 1.0
            if i >= tails_s and i <= tails_e:
                tails[i] = 1.0

        indexed_tokens = torch.tensor(indexed_tokens).long()  # (1, L)
        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[:avai_len] = 1
        print("heads",heads)
        print("heads len",len(heads))

        print("indexed_tokens", indexed_tokens)
        print("indexed_tokens  len ", len(indexed_tokens))
        print("att_mask",att_mask)

        # data2dic = {'word': word1,
        #             'token': relation,
        #             'h': {'pos':head_pos},
        #             't': {'pos':tail_pos}
        #
        #             }
        # # all_dict.append(data2dic)
        #
        # self.writeTxt(data2dic)
        # print(data2dic)

        return indexed_tokens, heads, tails, att_mask


    def _tokenize(self, tokens):
        re_tokens = ['[CLS]']
        for token in tokens.strip().split():
            re_tokens += self.berttokenizer.tokenize(token)
            # print('re_tokens',re_tokens)
            # print('token',token)
            # print('len',len(re_tokens))
        re_tokens.append('[SEP]')
        return re_tokens
    def find_head_idx(self, source, target):
        target_len = len(target)
        for i in range(len(source)):
            if source[i: i + target_len] == target:
                return i
        return -1

    def writeTxt(dic):
        with open("./veh_test2.txt", 'a') as fout:
            # print(all_dict[i])
            fout.write(json.dumps(dic))
            fout.write('\r\n')
            print("写入成功")

if __name__ == '__main__':
    config = Config()
    a = PicDataset(path=config.data_pic_path,rel2id_path=config.veh_rel2id,pretrain_path=config.bert_base)
    for k in a:
        print(k)
    # print(a.value)


