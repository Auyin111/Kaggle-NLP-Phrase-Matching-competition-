import os


class Cfg:

    def __init__(self):
        train = True
        seed = 42
        # TODO
        wandb = False
        competition = 'PPPM'
        _wandb_kernel = 'nakama'
        debug = False
        apex = True
        print_freq = 100

        bbbb = 10000000000000000

        class Dir:
            # TODO
            output = 'output'
            train_log = os.path.join(output, 'train_predict.log')

            data = 'data'

        class Train:
            num_workers = 4
            batch_size = 16
            scheduler = 'cosine'  # ['linear', 'cosine']
            batch_scheduler = True
            num_cycles = 0.5
            num_warmup_steps = 0
            epochs = 3
            max_len = 512

        class CV:
            n_fold = 4
            trn_fold = [0, 1, 2, 3]

        class Model:
            # TODO
            pretrained_model = "microsoft/deberta-v3-large"

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


if __name__ == '__main__':
    print(Cfg.Dir.train_log)
