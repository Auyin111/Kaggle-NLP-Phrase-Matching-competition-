from ensemble.ensemble_model import EnsembleModel
import category_encoders as ce
import os
import pandas as pd
import torch
import gc
import datetime
import importlib
from utils import highlight_string
from gen_data.dataset import TestDataset, find_max_len
from torch.utils.data import DataLoader
from train_predict.inference import inference_fn
from transformers import AutoTokenizer, DataCollatorWithPadding
from gen_data.dataset import create_text, merge_context


# TODO: combine with the predict_result in main.py
def predict_result(version, is_debug, device=None):
    # load the specific version cfg, model, AutoTokenizer, model parameter

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

    df_test = pd.read_csv(os.path.join(cfg.dir_data, 'test.csv'))

    df_test = merge_context(df_test, cfg)
    df_test = create_text(df_test, cfg.use_grp_2)

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

    predictions = []
    for fold in cfg.trn_fold:
        model = CustomModel(cfg,
                            config_path=os.path.join(cfg.dir_output, 'model.config'),
                            pretrained=False)
        # load specific model parameter
        state = torch.load(os.path.join(cfg.dir_output, 'model', f"fold_{fold}_best.model"),
                           map_location=torch.device(cfg.device))
        model.load_state_dict(state['model'])

        prediction = inference_fn(test_loader, model, cfg.device)
        predictions.append(prediction)
        del model, state, prediction;
        gc.collect()
        torch.cuda.empty_cache()

        print(predictions)


if __name__ == '__main__':

    list_em = ['rf', 'en']
    em_version = 'em1.0.1'
    is_debug = True
    # encoder = ce.BinaryEncoder()
    encoder = None
    list_model_version = ['v3.1.1', 'albert-base-v2', 'deberta-v3-base ver1']
    n_jobs = 10

    em_trainer = EnsembleModel(list_model_version, list_em, em_version, encoder, is_debug, n_jobs=n_jobs)
    em_trainer.find_best_model()
    # em_trainer.refit_em_model(best_model, dict_em_best_params,  list_model_version)

    em_trainer.find_best_em()