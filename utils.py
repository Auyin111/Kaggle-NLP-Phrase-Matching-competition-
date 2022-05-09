import scipy as sp
import numpy as np
import os, random, time, math
import torch


def get_logger(path):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s","%Y-%m-%d %H:%M:%S"))
    handler2 = FileHandler(filename=path)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score


def get_result(oof_df, cfg):
    labels = oof_df['score'].values
    preds = oof_df['pred'].values
    score = get_score(labels, preds)
    cfg.logger.info(f'Score: {score:<.4f}')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def highlight_string(string, symbol='-'):
    str_len = len(string)

    str_top_bottom = (str_len + 4) * symbol
    str_middle = f'{symbol} {string} {symbol}'
    str_output = f"""\n\n{str_top_bottom}\n{str_middle}\n{str_top_bottom}\n"""

    return str_output