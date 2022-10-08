import torch
import torchvision

from PIL import Image, ImageFilter
import numpy as np


def moco_nonsem_get_model(weigths_path, num_classes=128):
    model = torchvision.models.resnet50(num_classes=num_classes)
    state_dict = torch.load(weigths_path)["state_dict"]

    for k in list(state_dict.keys()):
        if k.startswith("module."):
            # remove prefix
            k_no_prefix = k[len("module.") :]
            state_dict[k_no_prefix] = state_dict[k]
            del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)

    normalized_model = torch.nn.Sequential(
        torchvision.transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model,
    )

    return normalized_model


moco_nonsem_preprocessing_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 244)),
        torchvision.transforms.ToTensor(),
    ]
)

moco_nonsem_inverse_transform = torchvision.transforms.Compose(
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


moco_nonsem_training_augmentation = torchvision.transforms.Compose(
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
