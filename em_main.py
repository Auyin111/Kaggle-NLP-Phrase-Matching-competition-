from ensemble.ensemble_model import EnsembleModel
import os
import joblib
import pandas as pd
import torch
import json
import gc
import datetime
import importlib
from utils import highlight_string
from gen_data.dataset import TestDataset, find_max_len
from torch.utils.data import DataLoader
from train_predict.inference import inference_fn
from transformers import AutoTokenizer, DataCollatorWithPadding
from gen_data.dataset import create_text, merge_context
from torch.optim.swa_utils import AveragedModel


# TODO: improve
def prepare_test_set(version, is_debug, device=None):
    """use fake version to find test set first"""

    # load specific version Cfg
    str_cfg_module = "output." + version + '.' + 'cfg'
    cfg_module = importlib.import_module(str_cfg_module)
    Cfg = cfg_module.Cfg

    cfg = Cfg(version, is_debug=is_debug, with_wandb=False, device=device)

    df_test = pd.read_csv(os.path.join(cfg.dir_data, 'test.csv'))

    df_test = merge_context(df_test, cfg)
    df_test = create_text(df_test, cfg.use_grp_2)

    return df_test


# TODO: combine with the predict_result in main.py
def predict_result(version, df_test, is_debug, device=None):
    """load the specific version cfg, model, AutoTokenizer, model parameter and make prediction"""

    # load specific version model
    str_model_module = "output." + version + '.' + 'model'
    model_module = importlib.import_module(str_model_module)
    CustomModel = model_module.CustomModel

    # load specific version Cfg
    str_cfg_module = "output." + version + '.' + 'cfg'
    cfg_module = importlib.import_module(str_cfg_module)
    Cfg = cfg_module.Cfg

    cfg = Cfg(version, is_debug=is_debug, with_wandb=False, device=device)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_status = highlight_string(f'predict by {cfg.version} model at {current_time}')
    cfg.logger.info(message_status)

    # load specific version tokenizer
    print(os.path.join(cfg.dir_output, 'tokenizer'))
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

    df_pred = pd.DataFrame()
    for fold in cfg.trn_fold:
        model_ = CustomModel(cfg,
                            config_path=os.path.join(cfg.dir_output, 'model.config'),
                            pretrained=False)
        if cfg.use_swa:
            model = AveragedModel(model_)
        else:
            model = model_

        # load specific model parameter
        state = torch.load(os.path.join(cfg.dir_output, 'model', f"fold_{fold}_best.model"),
                           map_location=torch.device(cfg.device))
        model.load_state_dict(state['model'])

        prediction = inference_fn(test_loader, model, cfg.device)

        df_pred.loc[:, f'{version}_fold_{fold}'] = [x[0] for x in prediction]

        del model, state, prediction;
        gc.collect()
        torch.cuda.empty_cache()

    return df_pred


def em_predict_result(em_trainer, df_test, list_model_version, list_fold, is_debug):

    df_all_pred = pd.DataFrame()
    for version in list_model_version:
        df_pred = predict_result(version, df_test, is_debug, device=None)
        df_all_pred = pd.concat([df_all_pred, df_pred], axis=1)

    str_best_model = json.load(open(os.path.join(em_trainer.em_cfg.dir_em_output, 'dict_best_model.json')))['best_model']

    df_fold_prediction = pd.DataFrame()
    for fold in list_fold:
        print(f'fold: {fold}')
        list_features = [col for col in df_all_pred if f'_fold_{fold}' in col]
        path_em_model = os.path.join(em_trainer.em_cfg.dir_em_output, str_best_model, f'fold_{fold}', 'model.joblib')
        em_model = joblib.load(path_em_model)

        df_fold_prediction.loc[:, fold] = em_model.predict(df_all_pred[list_features])

    df_test.loc[:, 'score'] = df_fold_prediction.mean(axis=1)
    df_test[['id', 'score']].to_csv('submission.csv', index=False)


if __name__ == '__main__':

    list_em = ['rf', 'en']
    em_version = 'submission_ver_1'
    is_debug = False
    encoder = None  # ce.BinaryEncoder()
    list_model_version = [
        'deberta_large_MSE_BS32_grp2short_v2head_Ep12',
        'deberta_base_MSE_BS64_grp2short_v2head_epoch20',
        'bert-for-patents_MSE_BS32_grp2short_v2head_epoch12',
        'roberta-base_MSE_BS32_grp2short_v2head_epoch12',
        'albert-base-v2'
    ]
    if is_debug:
        list_fold = ['0', '1']
    elif is_debug is False:
        list_fold = ['0', '1', '2', '3']
    n_jobs = 12

    # find the test set only
    df_test = prepare_test_set(list_model_version[0], is_debug, device=None)

    em_trainer = EnsembleModel(list_model_version, list_em, em_version, encoder, is_debug, n_jobs=n_jobs)
    # train model and save the string of best model
    em_trainer.find_best_model(list_fold)
    em_predict_result(em_trainer, df_test, list_model_version, list_fold, is_debug)