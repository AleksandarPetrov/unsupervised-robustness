import torch
import torchvision
import torch.nn as nn

from PIL import Image, ImageFilter
import numpy as np

from collections import OrderedDict


def pixpro_get_model(weigths_path, num_classes=None):

    # if we want to extract features, build the full model
    if num_classes is None:
        model = torchvision.models.resnet50(zero_init_residual=True)
        print(model)

        state_dict = torch.load(weigths_path)["model"]

        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith("module.encoder"):
                # remove prefix
                state_dict[k[len("module.encoder.") :]] = state_dict[k]
            # delete renamed or unused k
            # del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        model = torch.nn.Sequential(*(list(model.children())[:-1] + [nn.Flatten()]))
        print(model)

    # for linear evaluation:
    else:
        model = torchvision.models.resnet50(num_classes=num_classes)

        state_dict = torch.load(weigths_path)["model"]

        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith("module.encoder") and not k.startswith("module.encoder.fc"):
                # remove prefix
                state_dict[k[len("module.encoder.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    normalized_model = torch.nn.Sequential(
        torchvision.transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model,
    )

    return normalized_model


pixpro_preprocessing_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 244)),
        torchvision.transforms.ToTensor(),
    ]
)

pixpro_inverse_transform = torchvision.transforms.Compose(
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


pixpro_training_augmentation = torchvision.transforms.Compose(
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
