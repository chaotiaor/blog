import torch.utils.data
from lib.data.iter import Iter


class DataLoader:

    def __init__(self, opt):
        self.opt = opt
        self.data_set = Iter(opt)
        self.data_loader = None

    def load_data(self):
        self.data_loader = torch.utils.data.DataLoader(
            self.data_set,
            batch_size=self.opt.batch_size,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.num_threads)
        )
        return self.data_loader

    def reset(self):
        self.data_set.shuffle()

    def __len__(self):
        return len(self.data_set)










