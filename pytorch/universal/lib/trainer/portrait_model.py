import torch
from lib.loss.function import loss_kl, FocalLoss
from torch.autograd import Variable
from lib.trainer.base_model import Base


class PortraitModel(Base):
    def __init__(self, model, opt):
        super(PortraitModel, self).__init__(model, opt)
        self.loss_boundary_function = FocalLoss(opt.gamma, opt.alpha, opt.size_average)
        self.loss_mask = None
        self.loss_boundary = None
        self.loss_mask_ori = None
        self.loss_boundary_ori = None
        self.loss_stability_mask = None
        self.loss_stability_boundary = None
        self.edge = None

    def set_input(self, inputs):
        self.input_ori = inputs['input_ori'].to(self.device)
        self.input = inputs['input'].to(self.device)
        self.edge = inputs['edge'].to(self.device)
        self.label = inputs['mask'].to(self.device)

    def optimize_parameters(self):
        output_mask, output_edge = self.netG(self.input)
        self.loss_mask = self.loss_function(output_mask, self.label)
        self.loss_boundary = self.loss_boundary_function(output_edge, self.edge)
        output_mask_ori, output_edge_ori = self.netG(self.input_ori)
        self.loss_mask_ori = self.loss_function(output_mask_ori, self.label)
        self.loss_stability_mask = loss_kl(output_mask, Variable(output_mask_ori.data, requires_grad=False),
                                           self.temperature)
        self.loss_all = self.loss_mask + self.loss_boundary + self.loss_mask_ori + self.loss_stability_mask
        self.optimizer.zero_grad()
        self.loss_all.backward()
        self.optimizer.step()

    def step_verify(self):
        with torch.no_grad():
            self.output, output_edge = self.netG(self.input)
            self.loss_mask = self.loss_function(self.output, self.label)
            self.loss_boundary = self.loss_boundary_function(output_edge, self.edge)
            output_mask_ori, output_edge_ori = self.netG(self.input_ori)
            self.loss_mask_ori = self.loss_function(output_mask_ori, self.label)
            self.loss_stability_mask = loss_kl(self.output, Variable(output_mask_ori.data, requires_grad=False),
                                               self.opt.temperature)
            self.loss_all = self.loss_mask + self.loss_boundary + self.loss_mask_ori + self.loss_stability_mask




