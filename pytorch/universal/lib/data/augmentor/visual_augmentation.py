import cv2
import numpy as np
import random
from PIL import ImageEnhance
from lib.data.augmentor import set_ratio


def blur_aug(src, ksize=(3, 3)):
    for i in range(src.shape[2]):
        src[:, :, i] = cv2.GaussianBlur(src[:, :, i], ksize, 1.5)
    return src


def gray(src):
    if random.random() < set_ratio:
        return src
    g_img = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    src[:, :, 0] = g_img
    src[:, :, 1] = g_img
    src[:, :, 2] = g_img
    return src


def swap_change(src):
    if random.random() < set_ratio:
        return src

    a = [0, 1, 2]
    k = random.sample(a, 3)
    res = src.copy()
    res[:, :, 0] = src[:, :, k[0]]
    res[:, :, 1] = src[:, :, k[1]]
    res[:, :, 2] = src[:, :, k[2]]
    return res


def adjust_contrast(image, factor):
    mean = image.mean(axis=0).mean(axis=0)
    return _clip((image - mean) * factor + mean)


def adjust_brightness(image, delta):
    return _clip(image + delta * 255)


def adjust_hue(image, delta):
    image[..., 0] = np.mod(image[..., 0] + delta * 180, 180)
    return image


def adjust_saturation(image, factor):
    """ Adjust saturation of an image.
    Args
        image: Image to adjust.
        factor: An interval for the factor multiplying the saturation values of each pixel.
    """
    image[..., 1] = np.clip(image[..., 1] * factor, 0, 255)
    return image


def _clip(image):
    """
    Clip and convert an image to np.uint8.
    Args
        image: Image to clip.
    """
    return np.clip(image, 0, 255).astype(np.uint8)


def _uniform(val_range):
    return np.random.uniform(val_range[0], val_range[1])


class ColorDistort:

    def __init__(
            self,
            contrast_range=(0.4, 1),
            brightness_range=(-0.2, 0.2),
            hue_range=(-0.1, 0.1),
            saturation_range=(0.5, 0.8)
    ):
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.hue_range = hue_range
        self.saturation_range = saturation_range

    def __call__(self, image):
        if self.contrast_range is not None:
            contrast_factor = _uniform(self.contrast_range)
            image = adjust_contrast(image, contrast_factor)
        if self.brightness_range is not None:
            brightness_delta = _uniform(self.brightness_range)
            image = adjust_brightness(image, brightness_delta)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.hue_range is not None:
            hue_delta = _uniform(self.hue_range)
            image = adjust_hue(image, hue_delta)

        if self.saturation_range is not None:
            saturation_factor = _uniform(self.saturation_range)
            image = adjust_saturation(image, saturation_factor)

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image


def data_aug_color(image):
    if random.random() < set_ratio:
        return image
    random_factor = np.random.randint(4, 17) / 10.
    color_image = ImageEnhance.Color(image).enhance(random_factor)
    random_factor = np.random.randint(4, 17) / 10.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
    random_factor = np.random.randint(6, 15) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
    random_factor = np.random.randint(8, 13) / 10.
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)


def data_aug_blur(image):
    if random.random() < set_ratio:
        return image

    select = random.random()
    if select < 0.3:
        k_size = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (k_size, k_size), 0)
    elif select < 0.6:
        k_size = random.choice([3, 5])
        image = cv2.medianBlur(image, k_size)
    else:
        k_size = random.choice([3, 5])
        image = cv2.blur(image, (k_size, k_size))
    return image


def data_aug_noise(image):
    if random.random() < set_ratio:
        return image
    mu = 0
    sigma = random.random() * 20.0
    image = np.array(image, dtype=np.float32)
    image += np.random.normal(mu, sigma, image.shape)
    return _clip(image)


def show_edge(mask_ori, n):
    mask = mask_ori.copy()
    edge = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    if n == 2:
        ret, binary = cv2.threshold(np.uint8(mask) * 255, 127, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # RETR_EXTERNAL
        cv2.drawContours(edge, contours[0], -1, 1, 3)
    elif n == 3:
        # 获得类别1
        ret, binary = cv2.threshold(np.uint8(mask)*127, 128, 255, cv2.THRESH_TOZERO_INV)
        contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # RETR_EXTERNAL
        edge = cv2.drawContours(edge, contours[0], -1, 2, 1)
        # 获得类别2
        ret, binary = cv2.threshold(np.uint8(mask) * 127, 120, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # RETR_EXTERNAL
        edge = cv2.drawContours(edge, contours[0], -1, 1, 1)
    else:
        raise NotImplementedError('no such class')
    return edge


if __name__ == '__main__':
    """
    """
    




