import torch
import torchvision
import torch.nn as nn

from PIL import Image, ImageFilter
import numpy as np

from collections import OrderedDict


def resnet_get_model(weigths_path, num_classes=None):

    if num_classes is None:
        model = torchvision.models.resnet50(pretrained=True)

        # remove the fc-layer at the end
        model = torch.nn.Sequential(*(list(model.children())[:-1] + [nn.Flatten()]))
        print(model)

    # for linear evaluation:
    else:
        model = torchvision.models.resnet50(pretrained=True)

    normalized_model = torch.nn.Sequential(
        torchvision.transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model,
    )

    return normalized_model


resnet_preprocessing_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 244)),
        torchvision.transforms.ToTensor(),
    ]
)

resnet_inverse_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
    ]
)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


resnet_training_augmentation = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        torchvision.transforms.RandomApply(
            [torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],  # not strengthened
            p=0.8,
        ),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ]
)
