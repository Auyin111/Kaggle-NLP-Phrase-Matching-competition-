from utils import  get_logger, seed_everything
from cfg import Cfg

import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import shutil
import string
import pickle
import random
import joblib
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
print(f"torch.__version__: {torch.__version__}")
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
# TODO: what is it
# %env TOKENIZERS_PARALLELISM=true



import torch
from torch.utils.data import Dataset


def prepare_input(cfg, text):
    print('in prepare_input: !!!!!!!!!!!!')
    print([var for var in dir(cfg) if not var.startswith('__')])
    print(f'cfg.aaaa: {cfg.aaaa}')
    print('!!!!!!!!!!!!')



    # quit()

    inputs = cfg.tokenizer(text,
                           add_special_tokens=True,
                           max_length=cfg.Train.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):

        self.cfg = cfg

        print('init TrainDataset ##############')
        print(f'self.cfg.aaaa: {self.cfg.aaaa}')
        self.cfg.cccc = 100000000000
        print([var for var in dir(self.cfg) if not var.startswith('__')])

        print('###############')

        self.texts = df['text'].values
        self.labels = df['score'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        print('when get item ----------------------')
        print([var for var in dir(self.cfg) if not var.startswith('__')])
        print(f'cfg.aaaa: {self.cfg.aaaa}')
        print('-----------------------')

        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label

from cfg import Cfg

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    logger = get_logger(Cfg.Dir.train_log)
    seed_everything(seed=42)

    df_train = pd.read_csv(os.path.join(Cfg.Dir.data, 'train.csv'))
    df_test = pd.read_csv(os.path.join(Cfg.Dir.data, 'test.csv'))

    # preprocess
    # TODO: create text column
    df_train['text'] = df_train.anchor + '[SEP]' + df_train.target + ['SEP']
    df_test['text'] = df_test.anchor + '[SEP]' + df_test.target + ['SEP']

    df_train['score_map'] = df_train.score.map({0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1: 4})

    # cv split
    fold = StratifiedKFold(n_splits=Cfg.CV.n_fold, shuffle=True, random_state=Cfg.seed)

    for n, (train_index, val_index) in enumerate(fold.split(df_train, df_train['score_map'])):
        df_train.loc[val_index, 'fold'] = n
    df_train['fold'] = df_train.fold.astype(int)


    Cfg.aaaa = '1'
    Cfg.max_len = 100
    tokenizer = AutoTokenizer.from_pretrained(Cfg.Model.pretrained_model)
    tokenizer.save_pretrained(os.path.join(Cfg.Dir.output, 'tokenizer') )
    Cfg.tokenizer = tokenizer
    df = pd.DataFrame({'text':['1', '2', '3', '1', '2'] * 1000, 'score':[0,0,1,1,0]* 1000})

    train_dataset = TrainDataset(Cfg, df)
    train_loader = DataLoader(train_dataset,
                              batch_size=Cfg.Train.batch_size,
                              shuffle=True,
                              num_workers=Cfg.Train.num_workers, pin_memory=True, drop_last=True)


    print([var for var in dir(train_dataset.cfg) if not var.startswith('__')])


    for step, (inputs, labels) in enumerate(train_loader):
        print(f'inputs:::::: {inputs}')
        pass