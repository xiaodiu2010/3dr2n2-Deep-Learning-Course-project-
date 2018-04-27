import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
sys.path.append('../')
import cv2
from .img_aug import imgaug as ia
from .img_aug import augmenters as iaa


class compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class convertFromInts(object):
    def __call__(self, image, mask):
        return image.astype(np.float32), mask


class nomalize(object):
    def __init__(self, nomalizer):
        self.nomalizer = np.array(nomalizer, dtype=np.float32)

    def __call__(self, image, mask):
        image = image.astype(np.float32)
        image /= self.nomalizer
        image = 2*image - 1
        return image.astype(np.float32), mask


class augment_all(object):
    def __init__(self, ):
        self.st = lambda aug: iaa.Sometimes(0.5, aug)

        self.seq_affine = iaa.Sequential([
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_px={"x": (-10, 10), "y": (-10, 10)}, # translate by -16 to +16 pixels (per axis)
                rotate=(-10, 10), # rotate by -45 to +45 degrees
                order=1, # use any of scikit-image's interpolation methods
                cval=0., # if mode is constant, use a cval between 0 and 1.0
                mode="constant" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ),
            iaa.Fliplr(0.5, name="Fliplr"),  # horizontally flip 50% of all images
            #iaa.Flipud(0.5, name="Flipud"),  # vertically flip 50% of all images
            self.st(iaa.Crop(percent=(0, 0.1), name="Crop")),  # crop images by 0-10% of their height/width
        ],
            random_order=True  # do all of the above in random order
        )


    def __call__(self, image, mask):
        ## affine images and masks
        image = self.seq_affine.augment_images([image])[0]
        return image, mask


class augmentation(object):
    def __init__(self, config, is_train=True):
        self.mean = config.mean
        self.is_train = is_train
        self.img_size = (config.img_out_shape[0], config.img_out_shape[1])
        self.mask_size = config.mask_out_shape
        self.augment_train = compose([
            augment_all(),
            convertFromInts(),
            nomalize(255.)
        ])
        self.augment_test = compose([
            convertFromInts(),
            nomalize(255.)
        ])

    def __call__(self, img, mask):
        if self.is_train:
            return self.augment_train(img, mask)
        else:
            return self.augment_test(img, mask)