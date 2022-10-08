import os
from typing import Tuple, Any

import tqdm
import numpy as np
import scipy.spatial as spatial

import torch
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torchvision

from PIL import Image, ImageFont, ImageDraw

from typing import List
from dataclasses import dataclass, field


@dataclass
class AttackHistory:
    """
    Holds the norm of the perturbation and the divergence 
    in representation space at each iteration of the attack.
    """

    dist_l1: List[float] = field(default_factory=list)
    dist_l2: List[float] = field(default_factory=list)
    dist_linf: List[float] = field(default_factory=list)

    perturb_l1: List[float] = field(default_factory=list)
    perturb_l2: List[float] = field(default_factory=list)
    perturb_linf: List[float] = field(default_factory=list)

    cosine_similarity: List[float] = field(default_factory=list)


def unsqueeze_if_necessary(tensor):
    if len(tensor.size()) == 3:
        return torch.unsqueeze(tensor, 0)
    else:
        return tensor


class ImageFolderWithFilenames(ImageFolder):
    """
    Same as ImageFolder but instead of a class returns the name of the image file.
    """

    def load_without_transform(self, index: int) -> Any:
        path, target = self.samples[index]
        sample = self.loader(path)
        basename = os.path.basename(path)
        basename_wo_extension = os.path.splitext(basename)[0]

        return sample, basename_wo_extension

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, filaneme) where filename is the name of the image file.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        basename = os.path.basename(path)
        basename_wo_extension = os.path.splitext(basename)[0]

        return sample, basename_wo_extension

    def get_index_by_filename(self, filename: str) -> int:
        try:
            return next(i for i, (path, _) in enumerate(self.samples) if filename in path)
        except StopIteration:
            return FileNotFound(f"Filename {filename} not found in dataset!")


def analyze_and_show(
    model,
    target_rep_image,
    starting_image,
    inverted_image,
    inverse_transform,
    show=True,
    name_target="",
    name_starting="",
    img_prefix=None,
    csv_to_append=None,
):

    with torch.no_grad():
        rep_target = model(unsqueeze_if_necessary(target_rep_image))[0].detach().cpu()
        rep_starting_image = model(unsqueeze_if_necessary(starting_image))[0].detach().cpu()
        rep_inverse = model(unsqueeze_if_necessary(inverted_image))[0].detach().cpu()

    target_rep_image = unsqueeze_if_necessary(target_rep_image)[0]
    starting_image = unsqueeze_if_necessary(starting_image)[0]
    inverted_image = unsqueeze_if_necessary(inverted_image)[0]

    mse_rep_target_starting = torch.sqrt(torch.sum((rep_starting_image - rep_target) ** 2))
    mse_rep_target_inverse = torch.sqrt(torch.sum((rep_inverse - rep_target) ** 2))
    linf_rep_target_starting = torch.max(torch.abs(rep_starting_image - rep_target))
    linf_rep_target_inverse = torch.max(torch.abs(rep_target - rep_inverse))

    perturbation = inverted_image - starting_image

    l2_perturbation = nn.functional.mse_loss(perturbation, torch.zeros_like(perturbation))
    linf_perturbation = torch.max(torch.abs(perturbation))

    triplet = inverse_transform(torchvision.utils.make_grid([target_rep_image, inverted_image, starting_image], nrow=3))
    perturbation = torchvision.transforms.ToPILImage()(perturbation)
    images_to_show = Image.new("RGB", (triplet.width + perturbation.width, triplet.height))
    images_to_show.paste(triplet, (0, 0))
    images_to_show.paste(perturbation, (triplet.width + 1, 1))

    valid = torch.all(inverted_image >= 0.0) and torch.all(inverted_image <= 1.0)

    text_img = Image.new("RGB", (images_to_show.size[0], 40), color="black")
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(text_img)
    draw.text(
        (5, 5),
        fill="white",
        font=font,
        text=f"Rep.Init.Dist:   {mse_rep_target_starting:.8f}(L2avg) {linf_rep_target_starting:.8f}(Linf)",
    )
    draw.text(
        (5, 15),
        fill="white",
        font=font,
        text=f"Rep.Optim.Dist:  {mse_rep_target_inverse:.8f}(L2avg) {linf_rep_target_inverse:.8f}(Linf)\n",
    )
    draw.text(
        (5, 25),
        fill="white",
        font=font,
        text=f"PerurbationNorm: {l2_perturbation:.8f}(L2avg) {linf_perturbation:.8f}(Linf)",
    )
    draw.text(
        (405, 5),
        fill="white",
        font=font,
        text=f"Pixel value bounds: {'VALID' if valid else 'NOT VALID'}",
    )

    final_img = Image.new("RGB", (text_img.width, text_img.height + images_to_show.height))
    final_img.paste(text_img, (0, 0))
    final_img.paste(images_to_show, (0, text_img.height))

    if img_prefix is not None:
        path = f"{img_prefix}_{name_target}_{name_starting}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        final_img.save(path, "png")

    if csv_to_append is not None:
        os.makedirs(os.path.dirname(csv_to_append), exist_ok=True)
        df = pd.DataFrame.from_dict(
            {
                "target_rep_img": [name_target],
                "starting_img": [name_starting],
                "rep_init_dist_l2avg": [mse_rep_target_starting.item()],
                "rep_init_dist_linf": [linf_rep_target_starting.item()],
                "rep_optim_dist_l2avg": [mse_rep_target_inverse.item()],
                "rep_optim_dist_linf": [linf_rep_target_inverse.item()],
                "perturbation_norm_l2avg": [l2_perturbation.item()],
                "perturbation_norm_linf": [linf_perturbation.item()],
            }
        )
        df.to_csv(
            csv_to_append,
            mode="a",
            header=not os.path.exists(csv_to_append),
            index=False,
        )

    if show:
        print(
            f"Rep.Init.Dist:  \t{mse_rep_target_starting:.8f}(L2avg) \t{linf_rep_target_starting:.8f}(Linf)\n"
            f"Rep.Optim.Dist: \t{mse_rep_target_inverse:.8f}(L2avg) \t{linf_rep_target_inverse:.8f}(Linf)\n"
            f"PerurbationNorm: \t{l2_perturbation:.8f}(L2avg) \t{linf_perturbation:.8f}(Linf)"
        )

        print(f"Pixel value bounds: {'VALID' if valid else 'NOT VALID'}")

        display(final_img)
