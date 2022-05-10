import os
from utils import get_logger
from pathlib import Path


class Cfg:

    Path("output").mkdir(parents=True, exist_ok=True)

    def __init__(self):

        # dir_data
        if os.path.exists('/kaggle/input'):
            print('Reminder: you are running in Kaggle')
            self.dir_data = '/kaggle/input'
            self.on_kaggle = True
        else:
            print('Reminder: you are not in kaggle')
            self.dir_data = 'kaggle/input'
            self.on_kaggle = False

    version = 'v2.0.4'
    # dir and path
    dir_output = os.path.join('output', version)
    dir_own_dataset = 'own_dataset'

    train = True
    seed = 42
    # TODO
    with_wandb = True
    competition = 'PPPM'
    _wandb_kernel = 'nakama'
    debug = False
    apex = True
    print_freq = 100

    # training
    num_workers = 4
    batch_size = 16
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 3
    max_len = 128

    # CV
    n_fold = 4
    trn_fold = [0, 1, 2, 3]

    # Model
    # pretrained_model = "microsoft/deberta-v3-large"
    pretrained_model = "microsoft/deberta-v3-base"
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    fc_dropout = 0.2
    target_size = 1
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000

    # logger
    logger = get_logger(os.path.join('output', 'train.log'))

    # wandb
    user = 'rlin'
    notes = 'example'


if __name__ == '__main__':

    cfg = Cfg()
    print(cfg.version)
