import os
import wandb
import pandas as pd
import torch
import gc
import numpy as np
import warnings
from utils import seed_everything
from train_predict.train import train_loop
from sklearn.model_selection import StratifiedKFold
from utils import get_result
import datetime
from utils import highlight_string
from gen_data.dataset import TestDataset, find_max_len
from torch.utils.data import DataLoader
from model.model import CustomModel
from train_predict.inference import inference_fn
from transformers import AutoTokenizer, DataCollatorWithPadding
from cfg import Cfg
from gen_data.dataset import create_text, merge_context

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# TODO: study what is it
# %env TOKENIZERS_PARALLELISM=true


def train_model(version, with_wandb, is_debug=False, device=None, version_protection=True):
    # setting
    ####################################################
    cfg = Cfg(version, with_wandb, is_debug=is_debug, device=device)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_status = highlight_string(f'start train {cfg.version} model at {current_time} (is_debug: {cfg.is_debug})')
    cfg.logger.info(message_status)

    if cfg.with_wandb:
        try:
            wandb.init(project=f"patent_competition", entity="kaggle_winner",
                       group=f'{cfg.user}_{cfg.pretrained_model}', job_type="train",
                       name=cfg.version, notes=cfg.notes)
        except:
            print('can not connect to wandb, it may due to internet or secret problem')

    seed_everything(seed=42)

    # Conflict testing

    if version_protection == True:  # Allow disabling version protection for testing
        if not os.path.exists(cfg.dir_output):
            os.makedirs(cfg.dir_output)
        else:
            raise Exception(f'the version: {cfg.version} is used, please edit version before train model')
    else:
        protection_message = highlight_string("Version protection is off, make sure to turn it on for training submission models")
        cfg.logger.info(protection_message)

    if (cfg.swa_start > cfg.epochs) and cfg.use_swa:  # Check if swa starts before the last epoch ends.
        raise Exception("SWA is enabled but SWA starts after the last epoch.")

    if cfg.target_size != 1 and cfg.target_size != 5:
        raise Exception("Target size can only either be 1 or 5")

    if cfg.target_size == 5 and cfg.loss_fn != "CE":
        raise Exception("Softmax output can only work with CE loss!")

    if cfg.target_size == 1 and cfg.loss_fn == "CE":
        raise Exception("Cannot use cross entropy on single class output!")

    # start
    ####################################################

    pd.options.display.max_colwidth = 1000

    df_train = pd.read_csv(os.path.join(cfg.dir_data, 'train.csv'))
    df_train = merge_context(df_train, cfg)

    # Allow the use of translated (currently traditional Chinese only) training set, but doesn't seem helping =(
    if (cfg.pretrained_model == "microsoft/mdeberta-v3-base" or cfg.pretrained_model == "xlm-roberta-base") and cfg.use_translated_data is True:
        df_train_translated = pd.read_csv(os.path.join(cfg.dir_data, 'train_translated.csv'))
        df_train_translated = merge_context(df_train_translated, cfg, "zh-TW")
        df_train = pd.concat([df_train, df_train_translated]).reset_index(drop=True)
    if cfg.is_debug:
        df_train = df_train.sample(n=200).reset_index(drop=True)

    df_train = create_text(df_train, cfg.use_grp_2, cfg.use_mentioned_groups)
    print(f"Length of the training set: {len(df_train)}")
    print(df_train["text"][:10])

    # tokenizer
    cfg.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model)
    dir_tokenizer = os.path.join(cfg.dir_output, 'tokenizer')
    if not os.path.exists(dir_tokenizer):
        os.makedirs(dir_tokenizer)
    cfg.tokenizer.save_pretrained(dir_tokenizer)

    # find max_len
    list_col_text = ['anchor', 'target', 'text_grp_1']
    if cfg.use_grp_2:
        list_col_text.append('text_grp_2')
    if cfg.use_mentioned_groups:
        list_col_text.append('mentioned_groups_grp_2')
    cfg = find_max_len(cfg, df_train, list_col_text)

    df_train['score_map'] = df_train.score.map({0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1: 4})

    if cfg.target_size == 5:  # One hot label encoding if softmax classification
        df_train['score'] = df_train['score'].apply(lambda x: torch.nn.functional.one_hot(torch.tensor(int(x * 4)), num_classes=5))

    fold = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)

    for n, (train_index, val_index) in enumerate(fold.split(df_train, df_train['score_map'])):
        df_train.loc[val_index, 'fold'] = n
    df_train['fold'] = df_train.fold.astype(int)

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

        df_valid.to_csv(os.path.join(cfg.dir_output, 'df_valid.csv'), index=False)

    if cfg.with_wandb:
        wandb.finish()


def predict_result(version, is_debug, device=None):
    cfg = Cfg(version, is_debug=is_debug, with_wandb=False, device=device)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_status = highlight_string(f'predict by {cfg.version} model at {current_time}')
    cfg.logger.info(message_status)

    df_test = pd.read_csv(os.path.join(cfg.dir_data, 'test.csv'))

    df_test = merge_context(df_test, cfg)
    df_test = create_text(df_test, cfg.use_grp_2)

    cfg.tokenizer = AutoTokenizer.from_pretrained(os.path.join(cfg.dir_output, 'tokenizer'))

    # find max_len
    list_col_text = ['anchor', 'target', 'text_grp_1']
    if cfg.use_grp_2:
        list_col_text.append('text_grp_2')
    cfg = find_max_len(cfg, df_test, list_col_text)

    test_dataset = TestDataset(cfg, df_test)
    collator = DataCollatorWithPadding(cfg.tokenizer) if cfg.dynamic_padding else None
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=cfg.num_workers,
                             collate_fn=collator,
                             pin_memory=True, drop_last=False)

    predictions = []
    for fold in cfg.trn_fold:
        model = CustomModel(cfg,
                            config_path=os.path.join(cfg.dir_output, 'model.config'),
                            pretrained=False)
        state = torch.load(os.path.join(cfg.dir_output, 'model', f"fold_{fold}_best.model"),
                           map_location=torch.device(cfg.device))
        model.load_state_dict(state['model'])

        prediction = inference_fn(test_loader, model, cfg.device)
        predictions.append(prediction)
        del model, state, prediction;
        gc.collect()
        torch.cuda.empty_cache()

    df_test.loc[:, 'score'] = np.mean(predictions, axis=0)
    df_test[['id', 'score']].to_csv('submission.csv', index=False)


if __name__ == '__main__':

    version = "testing"
    version_protection = True
    is_debug = True
    if is_debug is True:
        version_protection = False

    train_model(version, False, is_debug=is_debug, version_protection=version_protection)
    predict_result(version, is_debug=is_debug, device='cuda')

    df = pd.read_csv(os.path.join('output', version, 'df_valid.csv'))
    # df.to_csv('output/v3.1.1/df_valid.csv')d
    print(df.fold.value_counts())
    print(df.head())
