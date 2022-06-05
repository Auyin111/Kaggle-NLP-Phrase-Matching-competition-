import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def prepare_input(cfg, text):  # Modified for dynamic padding
    inputs = cfg.tokenizer(text,
                            add_special_tokens=True,
                            truncation = True,
                            max_length = cfg.max_len,
                            padding = False if cfg.dynamic_padding else "max_length",
                            return_offsets_mapping = False)

    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


def find_max_len(cfg, df, list_col_text):
    """find the max length in the text"""

    dict_col_text = {}

    for col_text in list_col_text:

        if col_text not in dict_col_text:
            dict_col_text[col_text] = 0

        for text in tqdm(df[col_text].fillna("")):
            length = len(cfg.tokenizer(text, add_special_tokens=False)['input_ids'])

            if length > dict_col_text[col_text]:
                dict_col_text[col_text] = length

    max_len = 0
    for k, length in dict_col_text.items():
        max_len += length
    # CLS + SEP + SEP
    max_len += len(list_col_text)
    cfg.max_len = max_len

    cfg.logger.info(f'dict_col_text: {dict_col_text}')
    cfg.logger.info(f'cfg.max_len: {cfg.max_len}')

    return cfg


class TrainDataset(Dataset):
    def __init__(self, cfg, df):

        self.cfg = cfg
        self.texts = df['text'].values
        self.labels = df['score'].values

    def __len__(self):
        return len(self.labels)

    def get_labels(self,item):
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return label

    def __getitem__(self, item):

        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        input_dict = {"labels": label}
        for k, v in inputs.items():
            input_dict[k] = v
        return input_dict


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs