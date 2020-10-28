from PIL import Image
import numpy as np
import os
import shutil


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths, model):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path, model)
    else:
        mkdir(paths, model)


def mkdir(path, model):
    if model:
        if os.path.exists(path):
            shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


def label_color_map(n):
    if n == 3:
        cmap = np.array([(255,  0,  0), (0,  255,  0), (0,  0,  255)], dtype=np.uint8)
        return cmap


class Colorize(object):
    def __init__(self, n=3):
        self.n = n
        self.cmap = label_color_map(n)

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros([3, size[0], size[1]], dtype=np.uint8)

        for index, label in enumerate(range(self.n)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[index][0]
            color_image[1][mask] = self.cmap[index][1]
            color_image[2][mask] = self.cmap[index][2]
        color_image = color_image.transpose((1, 2, 0))
        return color_image


class ColorizeV2(object):
    def __init__(self, n=3):
        self.n = n
        self.cmap = label_color_map(n)

    def __call__(self, gray_image, color_image):
        color_image = color_image.transpose((2, 0, 1))
        for index, label in enumerate(range(1, self.n)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[index][0]
            color_image[1][mask] = self.cmap[index][1]
            color_image[2][mask] = self.cmap[index][2]
        color_image = color_image.transpose((1, 2, 0))
        return color_image


