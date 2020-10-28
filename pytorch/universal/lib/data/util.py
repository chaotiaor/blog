"""
主要是一些转化数据的辅助函数
"""
import torch
from torch.nn import functional as f
import torchvision.transforms as transforms
import numpy as np


def get_transform_sigmoid(label, n):
    """
    :param label:
    :param n:
    :return:
    """
    label = torch.from_numpy(label).long()
    label = f.one_hot(label, n)
    label = label.permute(2, 0, 1).float()
    return label


def get_transform_image():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.485, 0.456, 0.408),
                                            (0.231, 0.231, 0.231))
                       ]
    return transforms.Compose(transform_list)


def get_transform_other(label):
    """
    :param label:
    :return:
    """
    label = torch.from_numpy(label).long()
    return label


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    epsilon = 1e-5
    return (np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


if __name__ == '__main__':
    """
    
    """
    a1 = np.array([[[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]], [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]]])
    b1 = np.array([[[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0]], [[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0]]])
    print(a1.shape)
    print(b1.shape)
    # a1 = np.ones([1, 2, 2, 3], np.int)
    # b1 = np.zeros([1, 2, 2, 3], np.int)
    # a1 = np.argmax(a1, axis=-1)
    # b1 = np.argmax(b1, axis=-1)
    # a1 = a1.squeeze(0)
    # b1 = b1.squeeze(0)
    # print(a1.shape)
    cc = fast_hist(a1, b1, 3)
    # print(cc)
    # print(cc.sum(1))
    # print(cc.sum(0))
    # print(np.diag(cc))
    cc1 = per_class_iu(cc)
    print(cc1)
    # print(a1)
    # b1 = get_transform_sigmoid(a1, 3)
    # print(b1)
    # print(b1.shape)
    # c1 = np.array(b1).argmax(axis=0)
    # print(c1)














