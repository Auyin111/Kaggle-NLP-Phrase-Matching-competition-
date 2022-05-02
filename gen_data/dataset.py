import torch
from torch.utils.data import Dataset


def prepare_input(cfg, text):
    print('in prepare_input: !!!!!!!!!!!!')
    print([var for var in dir(cfg) if not var.startswith('__')])
    print(f'cfg.aaaa: {cfg.aaaa}')
    print('!!!!!!!!!!!!')



    # quit()

    inputs = cfg.tokenizer(text,
                           add_special_tokens=True,
                           max_length=cfg.Train.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):

        self.cfg = cfg

        # print('init TrainDataset ##############')
        # print(f'self.cfg.aaaa: {self.cfg.aaaa}')
        # self.cfg.cccc = 100000000000
        # print([var for var in dir(self.cfg) if not var.startswith('__')])
        #
        # print('###############')

        self.texts = df['text'].values
        self.labels = df['score'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        print('when get item ----------------------')
        print([var for var in dir(self.cfg) if not var.startswith('__')])
        print(f'cfg.aaaa: {self.cfg.aaaa}')
        print('-----------------------')

        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label
