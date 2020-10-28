"""

"""
from lib.core import create_net
from options.test_options import TestOptions
from lib.data.loader import DataLoader
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm


def main(model_path, save_path, target_label=0):

    """

    :param model_path:
    :param save_path:
    :param target_label:
    :return:
    """
    opt_train = TestOptions().parse()
    s_model = create_net(opt_train)
    dataset = DataLoader(opt_train)

    model_dict = torch.load(model_path)
    s_model.load_state_dict(model_dict)
    s_model.eval()
    
    p_save = []
    label_data = []
    mask_paths = []
    image_paths = []
    for i, data in enumerate(dataset.load_data()):
        mask, edge = s_model(data['input'])
        p_mask = mask
        p_mask = np.array(p_mask.data)
        p_save.append(p_mask.squeeze(0))
        label_data.append(np.array(data['mask'].data).squeeze(0))
        mask_paths.append(data['mask_path'])
        image_paths.append(data['image_path'])

  
    np.savez(save_path,
             # recall=np.array(recall_p),
             # accuracy=np.array(accuracy_t),
             p_save=np.array(p_save),
             label=np.array(label_data),
             mask_paths=np.array(mask_paths),
             image_paths=np.array(image_paths)
             )


if __name__ == '__main__':
    import os
    import time
    path = 'checkpoints/experiment_name_sf'
    target_path = './log/10-21/sf'
    target_label = 1
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'latest' in file:
                model_path = os.path.join(root, file)
                file = file.split('.')[0]
                save_path = os.path.join(target_path, '{}_recall_accuracy_prediction_label_{}_{}'.
                                         format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                                                target_label, file))
                main(model_path, save_path, target_label=target_label)



