import os
import pandas as pd
import torch
import gc
import numpy as np
import datetime
from utils import highlight_string
from gen_data.dataset import TestDataset
from torch.utils.data import DataLoader
from model.model import CustomModel
from train_predict.inference import inference_fn
from transformers import AutoTokenizer
from cfg import Cfg


def predict_result():

    cfg = Cfg()

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_status = highlight_string(f'predict by {cfg.version} model at {current_time}')
    cfg.logger.info(message_status)

    df_test = pd.read_csv(os.path.join(cfg.dir_data, 'test.csv'))
    df_context_grp_1 = pd.read_csv(os.path.join(cfg.dir_own_dataset, 'df_context_grp_1.csv'))
    # form text column
    df_test.loc[:, 'context_grp_1'] = df_test.context.apply(lambda x: x[0])
    df_test = df_test.merge(df_context_grp_1, how='left', on='context_grp_1')
    assert df_test.text_grp_1.isnull().sum() == 0, 'some of the context text are missing'
    # TODO: map the context grp2 and update the text column
    df_test['text'] = df_test.context_grp_1 + '[SEP]' \
                       + df_test.anchor + '[SEP]' \
                       + df_test.target + ['SEP']

    cfg.tokenizer = AutoTokenizer.from_pretrained(os.path.join(cfg.dir_output, 'tokenizer'))

    test_dataset = TestDataset(cfg, df_test)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    predictions = []
    for fold in cfg.trn_fold:
        model = CustomModel(cfg,
                            config_path=os.path.join(cfg.dir_output, 'config', 'model.config'),
                            pretrained=False)
        state = torch.load(os.path.join(cfg.dir_output, 'model', f"fold_{fold}_best.model"),
                           map_location=torch.device('cpu'))
        model.load_state_dict(state['model'])

        prediction = inference_fn(test_loader, model, 'cpu')
        predictions.append(prediction)
        del model, state, prediction;
        gc.collect()
        torch.cuda.empty_cache()

    df_test.loc[:, 'score'] = np.mean(predictions, axis=0)
    df_test[['id', 'score']].to_csv('submission.csv', index=False)


if __name__ == '__main__':

    predict_result()