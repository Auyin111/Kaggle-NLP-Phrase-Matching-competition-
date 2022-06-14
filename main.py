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
from transformers import AutoTokenizer
from cfg import Cfg
from gen_data.dataset import create_text, merge_context
import sherpa
import sherpa.algorithms.bayesian_optimization as bayesian_optimization

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# TODO: study what is it
# %env TOKENIZERS_PARALLELISM=true

def load_param_into_cfg(cfg, trial):
    # set parameter with trial
    cfg.criterion = trial.parameters['encoder_lr']
    cfg.criterion = trial.parameters['decoder_lr']
    cfg.criterion = trial.parameters['min_lr']
    cfg.criterion = trial.parameters['eps']
    cfg.criterion = trial.parameters['betas']
    cfg.criterion = trial.parameters['fc_dropout']
    cfg.criterion = trial.parameters['target_size']
    cfg.criterion = trial.parameters['weight_decay']
    cfg.criterion = trial.parameters['gradient_accumulation_steps']
    cfg.criterion = trial.parameters['max_grad_norm']


def k_fold_train_model_and_get_scores(cfg, df_train):
    scores = []
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
        scores.append(get_result(df_valid, cfg))

        df_valid.to_csv(os.path.join(cfg.dir_output, 'df_valid.csv'), index=False)
    if cfg.with_wandb:
        wandb.finish()
    return scores


def train_model(version, with_wandb, is_debug=False, device=None):

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

    # start
    ####################################################
    if not os.path.exists(cfg.dir_output):
        os.makedirs(cfg.dir_output)
    else:
        raise Exception(f'the version: {cfg.version} is used, please edit version before train model')

    df_train = pd.read_csv(os.path.join(cfg.dir_data, 'train.csv'))
    # TODO
    if cfg.is_debug:
        df_train = df_train.head(200)

    df_train = merge_context(df_train, cfg)
    df_train = create_text(df_train, cfg.use_grp_2)

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
    cfg = find_max_len(cfg, df_train, list_col_text)

    df_train['score_map'] = df_train.score.map({0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1: 4})

    # cv split
    fold = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)

    for n, (train_index, val_index) in enumerate(fold.split(df_train, df_train['score_map'])):
        df_train.loc[val_index, 'fold'] = n
    df_train['fold'] = df_train.fold.astype(int)

    df_train.score.hist()

    algorithm = bayesian_optimization.GPyOpt(max_concurrent=1, model_type='GP_MCMC', acquisition_type='EI_MCMC',
                                             max_num_trials=10)
    # parameters = [sherpa.Discrete('encoder_lr', [2, 50]),
    #               sherpa.Choice('criterion', ['gini', 'entropy']),
    #               sherpa.Continuous('max_features', [0.1, 0.9])]

    parameters = [sherpa.Continuous('encoder_lr', [2e-5, 10e-5]),
                  sherpa.Continuous('decoder_lr', [2e-5, 10e-5]),
                  sherpa.Continuous('min_lr', [1e-6, 10e-6]),
                  sherpa.Continuous('eps', [2e-5, 10e-5]),
                  sherpa.Continuous('betas', [2e-5, 10e-5]),
                  sherpa.Continuous('fc_dropout', [2e-5, 10e-5]),
                  sherpa.Continuous('target_size', [2e-5, 10e-5]),
                  sherpa.Continuous('weight_decay', [2e-5, 10e-5]),
                  sherpa.Continuous('gradient_accumulation_steps', [2e-5, 10e-5]),
                  sherpa.Continuous('max_grad_norm', [0.1, 0.9])]

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=False)
    for trial in study:
        cfg.logger.info(f"Trial {trial.id} with parameters {trial.parameters}" )
        load_param_into_cfg(cfg, trial)
        scores = k_fold_train_model_and_get_scores(cfg, df_train)
        cfg.logger.info(f"Scores: {scores}")
        study.add_observation(trial, iteration=1, objective=sum(scores)/len(scores))
        study.finalize(trial)
    cfg.logger.info(study.get_best_result())

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
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

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

    version = 'v4.1.6'
    is_debug = True
    train_model(version, with_wandb=True, is_debug=is_debug, device='cuda')
    predict_result(version, is_debug=is_debug,
                   # device='cpu'
                   )

    df = pd.read_csv(os.path.join('output', version, 'df_valid.csv'))
    print(df.fold.value_counts())
    print(df.head())