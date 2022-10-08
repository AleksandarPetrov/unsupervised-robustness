import torch
import torchvision

from PIL import Image, ImageFilter
import numpy as np


def moco_get_model(weigths_path, num_classes=128):
    model = torchvision.models.resnet50(num_classes=num_classes)
    state_dict = torch.load(weigths_path)["state_dict"]
    # dim_mlp = model.fc.weight.shape[1]
    # model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)

    if num_classes == 128:
        state_dict["fc.weight"] = state_dict["module.encoder_q.fc.2.weight"]
        state_dict["fc.bias"] = state_dict["module.encoder_q.fc.2.bias"]

        for k in [
            "module.encoder_q.fc.0.weight",
            "module.encoder_q.fc.0.bias",
            "module.encoder_q.fc.2.weight",
            "module.encoder_q.fc.2.bias",
        ]:
            del state_dict[k]

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
            # remove prefix
            k_no_prefix = k[len("module.encoder_q.") :]
            state_dict[k_no_prefix] = state_dict[k]  # leave encoder_q in the param name
            # print(f"Replacing {k} with {k_no_prefix}")
            # copy state from the query encoder into a new parameter for the key encoder
            state_dict[k_no_prefix.replace("encoder_q", "encoder_k")] = state_dict[k]
            del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    if num_classes != 128:
        assert set(msg.missing_keys) == {
            "fc.weight",
            "fc.bias",
        }, f"Missing keys: {msg.missing_keys}"

    normalized_model = torch.nn.Sequential(
        torchvision.transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model,
    )

    return normalized_model


moco_preprocessing_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 244)),
        torchvision.transforms.ToTensor(),
    ]
)

moco_inverse_transform = torchvision.transforms.Compose(
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


moco_training_augmentation = torchvision.transforms.Compose(
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
