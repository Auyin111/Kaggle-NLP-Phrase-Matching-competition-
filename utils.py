import scipy as sp
import numpy as np
import os, random, time, math
import torch
import stat
import shutil
import matplotlib.pyplot as plt


def cp_child_content(dir_source, dir_destination, list_cp_content):
    """copy child directory content"""

    for content in list_cp_content:

        path_source = os.path.join(dir_source, content)
        path_destination = os.path.join(dir_destination, content)

        if os.path.isdir(path_source):
            shutil.copytree(path_source,
                            path_destination)

        else:
            shutil.copy2(path_source, path_destination)


def copy_to_working_dir(source_dir, destination_dir):
    """copy all folder and file in working directory"""

    for folder_or_file in os.listdir(source_dir):

        path_folder_file = os.path.join(source_dir, folder_or_file)
        path_folder_file_des = os.path.join(destination_dir, folder_or_file)

        if os.path.isdir(path_folder_file):
            copy_and_overwrite(path_folder_file, path_folder_file_des)
        else:
            # copying the files to the
            # destination directory
            shutil.copy2(path_folder_file, destination_dir)


def copy_and_overwrite(from_path, to_path):

    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)


def rmtree(top):

    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(top)


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

# ====================================================
# Print out the distribution of labels/contexts in each fold
# ====================================================

def get_distribution(folds, cfg):
    if cfg.batch_distribution == "label" or cfg.batch_distribution == "context":

        if cfg.batch_distribution == "label":
            distribution = {s / 4: len(folds[folds["score"] == s / 4]) / len(folds) for s in range(5)}
            folds["distribution"] = folds["score"].map(distribution)

        elif cfg.batch_distribution == "context":
            distribution = folds["context"].apply(lambda x: x[0]).value_counts().apply(lambda x: x / len(folds)).to_dict()
            folds["distribution"] = folds["context"].apply(lambda x: x[0]).map(distribution)
        return distribution, folds["distribution"].tolist()

    return None

def plot_lr_fn(lrs_input):
    plt.plot(lrs_input)
    plt.show()