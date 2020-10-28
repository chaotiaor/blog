"""
"""
import os
import cv2
from PIL import Image
from lib.data.augmentor.visual_augmentation import data_aug_color, data_aug_blur, data_aug_noise, \
    show_edge, swap_change
from lib.data.augmentor.augmentation import rotate_aug, aug_matrix, data_aug_flip, random_crop, random_resize
from lib.data.util import get_transform_image, get_transform_sigmoid, get_transform_other
import random
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_data_set(numpy_data, image, label, is_train, balance_dict):
    """

    :param numpy_data:
    :param image:
    :param label:
    :param is_train:
    :param balance_dict:
    :return:
    """
    images, balance = [], []
    name_data = np.load(numpy_data)
    if is_train:
        for i in name_data:
            if i[2] == 'train':
                temp_name = [os.path.join(image, i[0]), os.path.join(label, i[0])]
                images.append(temp_name)
                for _ in range(balance_dict[i[1]]):
                    balance.append(temp_name)
    else:
        for i in name_data:
            if i[2] == 'test':
                temp_name = [os.path.join(image, i[0]), os.path.join(label, i[0])]
                images.append(temp_name)
                balance.append(temp_name)
    return images, balance


class Iter:

    def __init__(self, opt):
        self.opt = opt
        self.shape = opt.size
        self.w = opt.size[0]
        self.h = opt.size[1]
        self.classes = opt.n_class
        self.loss_type = opt.loss_type
        self.angle_range = opt.angle_range
        self.scale_range = opt.scale_range
        self.shrink = opt.shrink

        dir_image = os.path.join(opt.dataroot, 'images')
        dir_label = os.path.join(opt.dataroot, 'masks')
        numpy_path = os.path.join(opt.dataroot, 'data.npy')
        images, self.balance = make_data_set(numpy_path, dir_image, dir_label, opt.isTrain, opt.balance_dict)
        self.data_set_size = len(images)
        random.shuffle(self.balance)
        self.transform = get_transform_image()

    def shuffle(self):
        random.shuffle(self.balance)

    def __getitem__(self, item):
        image_path, label_path = self.balance[item]
        image = cv2.imread(image_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(label_path, 0)
        image = cv2.resize(image, self.shape)
        mask = cv2.resize(mask, self.shape, interpolation=cv2.INTER_NEAREST)
        # print(image.shape)

        if self.opt.isTrain is True:
            height, width, channel = image.shape
            # bbox = [0, 0, width - 1, height - 1]
            #
            img_aug, mask_aug = random_crop(image, mask, self.shrink, height, width)
            img_aug, mask_aug = random_resize(img_aug, mask_aug, height, width)
            #
            # h = aug_matrix(width, height, bbox, self.w, self.h,
            #                angle_range=self.angle_range, scale_range=self.scale_range, offset=int(self.h / 4))
            # img_aug, mask_aug = rotate_aug(h, image, mask, self.shape)
            img_aug_ori, mask_aug_ori, aug_flag = data_aug_flip(img_aug, mask_aug)
            #
            img_aug = Image.fromarray(img_aug_ori)
            img_aug = data_aug_color(img_aug)
            img_aug = np.asarray(img_aug)
            img_aug = data_aug_blur(img_aug)
            img_aug = data_aug_noise(img_aug)
            img_aug = swap_change(img_aug)

        else:
            # image = cv2.resize(image, self.shape)
            # mask = cv2.resize(mask, self.shape, interpolation=cv2.INTER_NEAREST)
            img_aug = np.copy(image)
            img_aug_ori = np.copy(image)
            mask_aug_ori = np.copy(mask)
        edge = show_edge(mask_aug_ori, self.classes)
        input_norm = self.transform(img_aug)
        input_ori_norm = self.transform(img_aug_ori)
        output_mask = np.array(mask_aug_ori)
        if self.loss_type == 'sigmoid' or self.loss_type == 'dice':
            output_mask = get_transform_sigmoid(output_mask, self.classes)
        else:
            output_mask = get_transform_other(output_mask)
        edge = get_transform_other(edge)
        input_dict = {
            'image_path': image_path, 'input_ori': input_ori_norm,
            'input': input_norm, 'edge': edge, 'mask': output_mask,
            'mask_path': label_path
        }
        return input_dict

    def __len__(self):
        return self.data_set_size


if __name__ == '__main__':
  """
  """

























