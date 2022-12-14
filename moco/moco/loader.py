# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter, Image
import random
import numpy as np


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GaussianNoise:
    """Gaussian noise augmentation."""

    def __init__(self, std=1):
        self.std = std

    def __call__(self, x):
        aug = x + np.random.normal(0, self.std, (x.size[1], x.size[0], 3))
        aug = np.clip(aug, 0, 255).astype(np.uint8)
        return Image.fromarray(aug)
