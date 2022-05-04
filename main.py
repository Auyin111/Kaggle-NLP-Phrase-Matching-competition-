from utils import  get_logger, seed_everything

from train_predict.train import train_loop
from utils import get_score
import os
# import wandb

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


import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from cfg import Cfg



# TODO: what is it
# %env TOKENIZERS_PARALLELISM=true

def get_result(oof_df, cfg):
    labels = oof_df['score'].values
    preds = oof_df['pred'].values
    score = get_score(labels, preds)
    cfg.logger.info(f'Score: {score:<.4f}')





if __name__ == '__main__':

    cfg = Cfg()
    # TODO: use kaggle api


    # remind: need convert to obj before pass in pytorch dataloader


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'device: {device}')
    seed_everything(seed=42)

    df_train = pd.read_csv(os.path.join(cfg.dir_data, 'train.csv'))
    print(df_train.shape)
    df_train = df_train.head(10)
    df_test = pd.read_csv(os.path.join(cfg.dir_data, 'test.csv'))

    # TODO: map the context and create text column
    df_train['text'] = df_train.anchor + '[SEP]' + df_train.target + ['SEP']
    df_test['text'] = df_test.anchor + '[SEP]' + df_test.target + ['SEP']

    df_train['score_map'] = df_train.score.map({0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1: 4})

    # cv split
    fold = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)

    for n, (train_index, val_index) in enumerate(fold.split(df_train, df_train['score_map'])):
        df_train.loc[val_index, 'fold'] = n
    df_train['fold'] = df_train.fold.astype(int)

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model)
    tokenizer.save_pretrained(os.path.join(cfg.dir_output, 'tokenizer'))
    cfg.tokenizer = tokenizer

    df_train.score.hist()
    # logger.info(f'b: 1')

    if cfg.train:
        oof_df = pd.DataFrame()
        for fold in range(cfg.n_fold):
            if fold in cfg.trn_fold:

                _oof_df = train_loop(df_train, fold, cfg)
                oof_df = pd.concat([oof_df, _oof_df])
                cfg.logger.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df, cfg)

        oof_df = oof_df.reset_index(drop=True)
        cfg.logger.info(f"========== CV ==========")
        get_result(oof_df, cfg)
        oof_df.to_pickle(cfg.dir_output + 'oof_df.pkl')

    # if cfg.with_wandb:
    #     wandb.finish()