import torch
from lib.trainer.base_model import Base


class NameModel(Base):
    def __init__(self, model, opt):
        super(NameModel, self).__init__(model, opt)

    def set_input(self, inputs):
        """这里把数据传入到设备中"""
       
    def optimize_parameters(self):
        """ 这里设置loss函数和更新模型权重"""
        
        self.optimizer.zero_grad()
        self.loss_all.backward()
        self.optimizer.step()

    def step_verify(self):
        """ 这里验证模型，不需要跟新权重"""
        with torch.no_grad():
            self.output = self.netG(self.input)
            self.loss_all = self.loss_function(self.output, self.label)






