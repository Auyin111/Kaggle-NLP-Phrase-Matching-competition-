import os
import torch
from utils import get_logger
from pathlib import Path


class Cfg:

    Path("output").mkdir(parents=True, exist_ok=True)

    def __init__(self, version, with_wandb, is_debug, device=None):

        self.version = version
        self.with_wandb = with_wandb
        self.dir_output = os.path.join('output', self.version)

        # dir_data
        if os.path.exists('/kaggle/input'):
            print('Reminder: you are running in Kaggle')
            self.dir_data = '/kaggle/input/us-patent-phrase-to-phrase-matching'
            self.on_kaggle = True
        else:
            print('Reminder: you are not in kaggle')
            self.dir_data = 'kaggle/input'
            self.on_kaggle = False

        self.is_debug = is_debug
        if self.is_debug:
            self.epochs = 2
            self.n_fold = 2
            self.trn_fold = [0, 1]
        else:
            self.epochs = 5
            # CV
            self.n_fold = 4
            self.trn_fold = [0, 1, 2, 3]

        self.device = device
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

    # determine whether to use grp 2 context
    use_grp_2 = True

    # dir and path
    dir_own_dataset = 'own_dataset'

    # wandb
    user = 'rlin'
    notes = 'example'
    _wandb_kernel = 'nakama'

    train = True
    seed = 42
    competition = 'PPPM'
    debug = False
    apex = True
    print_freq = 100

    # training
    batch_scheduler = True
    plot_lr = True  # (New)

    # Batch loader (New)
    batch_size = 16
    num_workers = 4
    dynamic_padding = True
    batch_distribution = 'context'  # ['label', 'context', None]

    # Model
    pretrained_model = "albert-base-v2"  # "microsoft/deberta-v3-base", "albert-base-v2"
    target_size = 1  # Only in original model

    # Optimizer
    scheduler='cosine'  # ['linear', 'cosine', 'cosine_annealing']
    num_cycles = 0.5  # For 'linear' or 'cosine' only
    num_warmup_steps = 1000  # For 'linear' or 'cosine' only
    encoder_lr = 3e-5
    decoder_lr = 3e-5
    min_lr = 3e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000

    # Stochastic weight average (New)
    use_swa = False
    early_stopping = True
    swa_start = 8  # swa starts at this epoch
    swa_lr = 1e-5
    anneal_steps = 800

    # Multi sample dropout (New)
    dropout_prop = 0.2
    dropout_sample_num = 4  # Number of multi sample dropout, 1 = no MSD
    msd_average = False  # Use average output instead of sum output in multi sample dropout

    # Self-Attention (New)
    self_attention_head_num = 1
    self_attention_dropout_prob = 0.1

    # logger
    logger = get_logger(os.path.join('output', 'train.log'))


if __name__ == '__main__':

    cfg = Cfg()
    print(cfg.version)
