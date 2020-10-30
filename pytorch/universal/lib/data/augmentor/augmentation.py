import numpy as np
import cv2
import random
import math
from lib.data.augmentor import PIXEL_MEAN, set_ratio


def rotate_aug(h, image, mask, shape):
    """

    :param h:
    :param image:
    :param mask:
    :param shape:
    :return:
    """
    image_aug = cv2.warpAffine(np.uint8(image), h, shape,
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=PIXEL_MEAN)

    mask_aug = cv2.warpAffine(np.uint8(mask), h, shape,
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT)

    return image_aug, mask_aug


def data_aug_flip(image, mask):
    if random.random() < set_ratio:
        return image, mask, False
    image = cv2.flip(image, 0)
    mask = cv2.flip(mask, 0)
    return image, mask, True


def random_crop(src, mask, shrink, h, w):
    if random.random() < 0.2:
        return src, mask

    h_shrink = int(h * shrink)
    w_shrink = int(w * shrink)
    b_src = cv2.copyMakeBorder(src, h_shrink, h_shrink, w_shrink, w_shrink,
                               borderType=cv2.BORDER_CONSTANT,
                               value=PIXEL_MEAN
                               )
    b_mask = cv2.copyMakeBorder(mask, h_shrink, h_shrink, w_shrink, w_shrink,
                                borderType=cv2.BORDER_CONSTANT,
                                value=(0, )
                                )

    start_h = random.randint(0, 2*h_shrink)
    start_w = random.randint(0, 2*w_shrink)

    target_img = b_src[start_h:start_h+h, start_w:start_w+w, :]
    target_mask = b_mask[start_h:start_h+h, start_w:start_w+w]

    return target_img, target_mask


def random_resize(src, mask, h, w):
    if random.random() < set_ratio:
        return src, mask
    try:
        b_mask = np.copy(mask)
        b_src = np.copy(src)
        wheres = (b_mask == 1)
        x = np.argmax(wheres, axis=0)
        x_ = np.where(x > 0)
        x_0 = x_[0][0]
        x_1 = x_[0][-1]
        y = np.argmax(wheres, axis=1)
        y_ = np.where(y > 0)
        y_0 = y_[0][0]
        y_1 = y_[0][-1]
        target_mask = b_mask[y_0:y_1, x_0:x_1]
        target_img = b_src[y_0:y_1, x_0:x_1, :]
        target_mask = cv2.resize(target_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        target_img = cv2.resize(target_img, (w, h))
    except Exception:
        return src, mask
    else:
        return target_img, target_mask


def aug_matrix(img_w, img_h, bbox, w, h, angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=40):
    """
    first Translation, then rotate, final scale.
        [sx, 0, 0]       [cos(theta), -sin(theta), 0]       [1, 0, dx]       [x]
        [0, sy, 0] (dot) [sin(theta),  cos(theta), 0] (dot) [0, 1, dy] (dot) [y]
        [0,  0, 1]       [         0,           0, 1]       [0, 0,  1]       [1]
    :param img_w:
    :param img_h:
    :param bbox:
    :param w:
    :param h:
    :param angle_range:
    :param scale_range:
    :param offset:
    :return:
    """
    # ratio = 1.0 * (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (img_w * img_h)
    x_offset = (random.random() - 0.5) * 2 * offset
    y_offset = (random.random() - 0.5) * 2 * offset
    dx = (w - (bbox[2] + bbox[0])) / 2.0
    dy = (h - (bbox[3] + bbox[1])) / 2.0

    matrix_trans = np.array([[1.0, 0, dx],
                             [0, 1.0, dy],
                             [0, 0, 1.0]])

    angle = random.random() * (angle_range[1] - angle_range[0]) + angle_range[0]
    scale = random.random() * (scale_range[1] - scale_range[0]) + scale_range[0]
    scale *= np.mean([float(w) / (bbox[2] - bbox[0]), float(h) / (bbox[3] - bbox[1])])
    alpha = scale * math.cos(angle / 180.0 * math.pi)
    beta = scale * math.sin(angle / 180.0 * math.pi)

    center_x = w / 2.0 + x_offset
    center_y = h / 2.0 + y_offset
    h = np.array([[alpha, beta, (1 - alpha) * center_x - beta * center_y],
                  [-beta, alpha, beta * center_x + (1 - alpha) * center_y],
                  [0, 0, 1.0]]
                 )

    h = h.dot(matrix_trans)[0:2, :]
    return h


if __name__ == '__main__':
    """
    
    """
  


