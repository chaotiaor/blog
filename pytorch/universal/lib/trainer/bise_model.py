import torch
from lib.loss.function import loss_kl
from torch.autograd import Variable
from lib.trainer.base_model import Base


class BiSeModel(Base):
    def __init__(self, model, opt):
        super(BiSeModel, self).__init__(model, opt)

    def set_input(self, inputs):
        self.input_ori = inputs['input_ori'].to(self.device)
        self.input = inputs['input'].to(self.device)
        self.label = inputs['mask'].to(self.device)

    def optimize_parameters(self):
        """ 这里设置loss函数和更新模型权重"""
        output, output_sup1, output_sup2 = self.netG(self.input)
        output_ori, output_sup1_ori, output_sup2_ori = self.netG(self.input_ori)
        loss1 = self.loss_function(output, self.label)
        loss2 = self.loss_function(output_sup1, self.label)
        loss3 = self.loss_function(output_sup2, self.label)
        loss_stability_1 = loss_kl(output, Variable(output_ori.data, requires_grad=False), self.temperature)
        loss_stability_2 = loss_kl(output_sup1, Variable(output_sup1_ori.data, requires_grad=False), self.temperature)
        loss_stability_3 = loss_kl(output_sup2, Variable(output_sup2_ori.data, requires_grad=False), self.temperature)
        self.loss_all = loss1 + loss2 + loss3 \
                        + loss_stability_1 + loss_stability_2 + loss_stability_3
        self.optimizer.zero_grad()
        self.loss_all.backward()
        self.optimizer.step()

    def step_verify(self):
        with torch.no_grad():
            self.output = self.netG(self.input)
            self.loss_all = self.loss_function(self.output, self.label)






