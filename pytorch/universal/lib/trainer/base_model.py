import os
import torch
from abc import ABC
from lib.loss.function import get_scheduler
from lib.loss.function import SigmoidLoss, FocalLoss, DiceLoss
from lib.data.util import fast_hist, per_class_iu
import numpy as np
import torch.nn as nn


class Base(ABC):
    def __init__(self, model, opt):
        self.netG = model
        self.opt = opt
        self.n_class = opt.n_class
        self.loss_type = opt.loss_type
        self.temperature = opt.temperature
        self.name = opt.model_name
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.loss_names = ['all']
        self.metric = opt.metric
        self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                                          weight_decay=opt.weight_decay)
        self.scheduler = get_scheduler(self.optimizer, opt)
        if opt.loss_type == 'sigmoid':
            self.loss_function = SigmoidLoss()
        elif opt.loss_type == 'softmax':
            self.loss_function = nn.CrossEntropyLoss()
        elif opt.loss_type == 'focal':
            self.loss_function = FocalLoss(opt.gamma, opt.alpha, opt.size_average)
        else:
            self.loss_function = DiceLoss()

        self.input_ori = None
        self.input = None
        self.label = None
        self.loss_all = None
        self.output = None

    def setup(self, opt):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if len(self.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.netG.to(self.gpu_ids[0])
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)

        self.print_networks(True)

    def set_input(self, inputs):
        pass

    def optimize_parameters(self):
        pass

    def train(self):
        self.netG.train()

    def eval(self):
        self.netG.eval()

    def step_verify(self):
        pass

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizer.param_groups[0]['lr']
        if self.opt.lr_policy == 'plateau':
            self.scheduler.step(self.metric)
        else:
            self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_losses(self):
        """Return traning losses / errors.
        train.py will print out these errors on console,
        and save them to a file"""
        return float(getattr(self, 'loss_' + self.loss_names[0]))

    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """

        if isinstance(self.name, str):
            save_filename = '%s_net_%s.pth' % (epoch, self.name)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, 'net' + self.name)
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.module.cpu().state_dict(), save_path)
                net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """

        if isinstance(self.name, str):
            load_filename = '%s_net_%s.pth' % (epoch, self.name)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, 'net' + self.name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            try:
                state_dict = torch.load(load_path, map_location=str(self.device))
                net.load_state_dict(state_dict)
            except FileNotFoundError as e:
                print(e)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')

        if isinstance(self.name, str):
            net = getattr(self, 'net' + self.name)
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            print('[Network %s] Total number of parameters : %.3f M' % (self.name, num_params / 1e6))
        print('-----------------------------------------------')






