import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import highlight_string

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


def merge_context(df, cfg, language=None):
    if language is None:
        df_context_grp_1 = pd.read_csv(os.path.join(cfg.dir_own_dataset, 'df_context_grp_1.csv'))
    else:
        df_context_grp_1 = pd.read_csv(os.path.join(cfg.dir_own_dataset, f'df_context_grp_1_{language.lower()}.csv'))
    # form text column
    df.loc[:, 'context_grp_1'] = df.context.apply(lambda x: x[0])
    df = df.merge(df_context_grp_1, how='left', on='context_grp_1')
    #assert df.text_grp_1.isnull().sum() == 0, 'some of the context text are missing'

    if cfg.use_grp_2:
        if language is None:
            df_context_grp_2 = pd.read_csv(os.path.join(cfg.dir_own_dataset, 'df_context_grp_2.csv'))
        else:
            df_context_grp_2 = pd.read_csv(os.path.join(cfg.dir_own_dataset, f'df_context_grp_2_{language.lower()}.csv'))

        df = df.merge(
            df_context_grp_2.rename(columns={'description': 'text_grp_2', 'mentioned_groups': 'mentioned_groups_grp_2'}),
            how='left', on='context')

        # check the missing of grp_2 context
        ar_missing_grp_2 = df[df['text_grp_2'].isnull()].context.unique()
        if len(ar_missing_grp_2) > 0:

            message = highlight_string(f'the following grp 2 are missing: {ar_missing_grp_2}', '!')
            cfg.logger.warning(message)

            message = highlight_string(f"missing rate:\n{df[['text_grp_2', 'mentioned_groups_grp_2']].isnull().mean()}",
                                       '!')
            cfg.logger.warning(message)
        else:
            message = highlight_string('Data quality checking: you are using grp_2 context and it do not have any missing')
            cfg.logger.info(message)

    return df

def create_text(df, use_grp_2=True, use_mentioned_groups=False):

    df['text'] = df.anchor + "[SEP]" + df.target + "[SEP]" + df.text_grp_1
    if use_grp_2:
        # only + "[SEP]" and text_grp_2 if text_grp_2 is not null
        df.loc[:, 'text'] = np.where(df.text_grp_2.isnull(), df.text,
                                     df.text + "; " + df.text_grp_2)  # Treat group1 and group2 context as same text, separating by "; " insteadt of "[SEP]"
    if use_mentioned_groups:
        df.loc[:, 'text'] = np.where(df.mentioned_groups_grp_2.isnull(), df.text, df.text + "[SEP]" + df.mentioned_groups_grp_2)

    return df


class TrainDataset(Dataset):  # Modified for dynamic padding
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
