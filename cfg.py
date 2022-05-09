import os
from utils import get_logger


class Cfg:

    version = 'v2.0.0'
    # dir and path
    dir_output = os.path.join('output', version)
    # TODO: kaggle? local?
    dir_data = 'data'

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
    print(cfg.wandb)
