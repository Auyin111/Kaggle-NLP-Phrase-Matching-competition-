import os
import wandb
import pandas as pd
import torch
import warnings
from utils import seed_everything
from train_predict.train import train_loop
from transformers import AutoTokenizer
from cfg import Cfg
from sklearn.model_selection import StratifiedKFold
from utils import get_result
import datetime
from utils import highlight_string

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# TODO: study what is it
# %env TOKENIZERS_PARALLELISM=true


def train_model():

    # setting
    ####################################################
    cfg = Cfg()

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_status = highlight_string(f'start train {cfg.version} model at {current_time}')
    cfg.logger.info(message_status)

    # TODO: use kaggle wandb api
    wandb.init(project=f"patent_competition", entity="kaggle_winner",
               group=f'{cfg.user}_{cfg.pretrained_model}', job_type="train",
               name=cfg.version, notes=cfg.notes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {device}')
    seed_everything(seed=42)

    # start
    ####################################################
    if not os.path.exists(cfg.dir_output):
        os.makedirs(cfg.dir_output)
    else:
        raise Exception(f'the version: {cfg.version} is used, please edit version before train model')

    df_train = pd.read_csv(os.path.join(cfg.dir_data, 'train.csv'))
    df_train = df_train.head(40)
    # TODO: map the context and update the text column
    df_train['text'] = df_train.anchor + '[SEP]' + df_train.target + ['SEP']

    df_train['score_map'] = df_train.score.map({0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1: 4})

    # cv split
    fold = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)

    for n, (train_index, val_index) in enumerate(fold.split(df_train, df_train['score_map'])):
        df_train.loc[val_index, 'fold'] = n
    df_train['fold'] = df_train.fold.astype(int)

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model)
    dir_tokenizer = os.path.join(cfg.dir_output, 'tokenizer')
    if not os.path.exists(dir_tokenizer):
        os.makedirs(dir_tokenizer)
    tokenizer.save_pretrained(dir_tokenizer)
    cfg.tokenizer = tokenizer

    df_train.score.hist()

    if cfg.train:
        df_valid = pd.DataFrame()
        for fold in range(cfg.n_fold):
            if fold in cfg.trn_fold:

                _df_valid = train_loop(df_train, fold, cfg)
                df_valid = pd.concat([df_valid, _df_valid])
                cfg.logger.info(f"========== fold: {fold} result ==========")
                get_result(_df_valid, cfg)

        df_valid = df_valid.reset_index(drop=True)
        cfg.logger.info(f"========== CV ==========")
        get_result(df_valid, cfg)

        dir_df_valid = os.path.join(cfg.dir_output, 'df_valid')
        if not os.path.exists(dir_df_valid):
            os.makedirs(dir_df_valid)
        df_valid.to_pickle(os.path.join(dir_df_valid, 'df.pkl'))

    if cfg.with_wandb:
        wandb.finish()


if __name__ == '__main__':

    train_model()