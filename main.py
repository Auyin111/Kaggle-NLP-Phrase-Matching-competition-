from utils import  get_logger, seed_everything
from cfg import Cfg
from train_predict.train import train_loop
from utils import get_score
import os
import wandb
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

import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

import tokenizers
import transformers
# print(f"torch.__version__: {torch.__version__}")
# print(f"tokenizers.__version__: {tokenizers.__version__}")
# print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
# TODO: what is it
# %env TOKENIZERS_PARALLELISM=true

def get_result(oof_df):
    labels = oof_df['score'].values
    preds = oof_df['pred'].values
    score = get_score(labels, preds)
    logger.info(f'Score: {score:<.4f}')


if __name__ == '__main__':

    from cfg import Cfg

    Cfg = Cfg()

    Cfg.dddd = 1111111111
    from gen_data.dataset import TrainDataset

    Cfg.aaaa = '1'
    Cfg.max_len = 100
    tokenizer = AutoTokenizer.from_pretrained(Cfg.Model.pretrained_model)
    tokenizer.save_pretrained(os.path.join(Cfg.Dir.output, 'tokenizer'))
    Cfg.tokenizer = tokenizer
    df = pd.DataFrame({'text': ['1', '2', '3', '1', '2'] * 1000, 'score': [0, 0, 1, 1, 0] * 1000})

    train_dataset = TrainDataset(Cfg, df)
    train_loader = DataLoader(train_dataset,
                              batch_size=Cfg.Train.batch_size,
                              shuffle=True,
                              num_workers=Cfg.Train.num_workers, pin_memory=True, drop_last=True)

    # print([var for var in dir(train_dataset.cfg) if not var.startswith('__')])

    for step, (inputs, labels) in enumerate(train_loader):
        print(f'inputs:::::: {inputs}')
        pass


    stop

    from cfg import Cfg
    from gen_data.dataset import TrainDataset

    Cfg.aaaa = '1'
    Cfg.max_len = 100
    tokenizer = AutoTokenizer.from_pretrained(Cfg.Model.pretrained_model)
    tokenizer.save_pretrained(os.path.join(Cfg.Dir.output, 'tokenizer'))
    Cfg.tokenizer = tokenizer
    df = pd.DataFrame({'text': ['1', '2', '3', '1', '2'], 'score': [0, 0, 1, 1, 0]})

    train_dataset = TrainDataset(Cfg, df)
    train_loader = DataLoader(train_dataset,
                              batch_size=Cfg.Train.batch_size,
                              shuffle=True,
                              num_workers=Cfg.Train.num_workers, pin_memory=True, drop_last=True)

    print([var for var in dir(Cfg) if not var.startswith('__')])
    print([var for var in dir(train_dataset.cfg2) if not var.startswith('__')])

    # for train_index, test_index in train_dataset:
    #     pass
    for step, (inputs, labels) in enumerate(train_loader):
        print(f'inputs:::::: {inputs}')
        pass

    stop


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'device: {device}')
    logger = get_logger(Cfg.Dir.train_log)
    seed_everything(seed=42)

    df_train = pd.read_csv(os.path.join(Cfg.Dir.data, 'train.csv'))
    df_test = pd.read_csv(os.path.join(Cfg.Dir.data, 'test.csv'))

    # TODO: map the context and create text column
    df_train['text'] = df_train.anchor + '[SEP]' + df_train.target + ['SEP']
    df_test['text'] = df_test.anchor + '[SEP]' + df_test.target + ['SEP']

    df_train['score_map'] = df_train.score.map({0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1: 4})

    # cv split
    fold = StratifiedKFold(n_splits=Cfg.CV.n_fold, shuffle=True, random_state=Cfg.seed)

    for n, (train_index, val_index) in enumerate(fold.split(df_train, df_train['score_map'])):
        df_train.loc[val_index, 'fold'] = n
    df_train['fold'] = df_train.fold.astype(int)

    tokenizer = AutoTokenizer.from_pretrained(Cfg.Model.pretrained_model)
    tokenizer.save_pretrained(os.path.join(Cfg.Dir.output, 'tokenizer'))
    Cfg.tokenizer = tokenizer
    Cfg.aaaa = '1111'

    # print(dir(Cfg))
    print(Cfg.aaaa)
    # print(Cfg.tokenizer)
    # print('###############')

    # quit()


    df_train.score.hist()
    # logger.info(f'b: 1')




    if Cfg.train:
        oof_df = pd.DataFrame()
        for fold in range(Cfg.CV.n_fold):
            if fold in Cfg.CV.trn_fold:
                _oof_df = train_loop(df_train, fold, Cfg, logger)
                oof_df = pd.concat([oof_df, _oof_df])
                logger.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        logger.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(Cfg.Dir.output + 'oof_df.pkl')

    if Cfg.wandb:
        wandb.finish()