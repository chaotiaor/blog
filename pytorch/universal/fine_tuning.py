"""
加载已有的模型进行微调
"""
import torch
from lib.core import create_net
from options.train_options import TrainOptions

opts = TrainOptions().parse()
model = create_net(opts)
torch.save(model.cpu().state_dict(), 'model.pth')

model_path = ''
model_dict = torch.load(model_path, map_location=torch.device('cpu'))

new_model_path = 'model.pth'
new_model_dict = torch.load(new_model_path, map_location=torch.device('cpu'))

for k, v in model_dict.items():
    if k in new_model_dict.keys():
        if new_model_dict[k].shape != v.shape:
            continue
        new_model_dict[k] = v

model.load_state_dict(new_model_dict)
torch.save(model.cpu().state_dict(), 'latest_net_G.pth')




