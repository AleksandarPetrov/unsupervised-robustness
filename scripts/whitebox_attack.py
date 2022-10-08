#!/usr/bin/env python3

import sys
import os
import yaml
import torch
import torchvision
import torch.backends.cudnn as cudnn
import numpy as np
import datetime
import tqdm
import random

import argparse

sys.path.append(".")
sys.path.append("..")
sys.path.append("../models")
sys.path.append("models")

from models import get_model

from utils import (
    ImageFolderWithFilenames,
    pgd_attack,
    analyze_and_show,
)


def cli(args=None):

    parser = argparse.ArgumentParser(description="Optimize origin to have a representation matching target.")
    parser.add_argument(
        "--origin",
        "-o",
        dest="origin",
        default=None,
        help="Starting image filename to optimize. If not provided, we will sample N images",
    )
    parser.add_argument(
        "--target",
        "-t",
        dest="target",
        default=None,
        help="Target image filename to optimize. If not provided, we will sample N images",
    )
    parser.add_argument(
        "--n-samples",
        dest="n_samples",
        type=int,
        help="Number of images to attack if origin and target are not provided.",
    )
    parser.add_argument(
        "--model",
        "-m",
        dest="model_name",
        required=True,
        help="Which model we use.",
    )
    parser.add_argument(
        "--model-weights",
        dest="model_weights_path",
        default=None,
        help="Path to the model weights.",
    )
    parser.add_argument(
        "--dataset",
        "-D",
        dest="dataset_path",
        required=True,
        help="Directory containing the dataset to use.",
    )
    targeted_group = parser.add_mutually_exclusive_group(required=True)
    targeted_group.add_argument("--targeted", action="store_true")
    targeted_group.add_argument("--untargeted", action="store_true")
    parser.add_argument(
        "--attack",
        "-a",
        dest="attack_type",
        required=True,
        default="pgd",
        choices=["pgd"],
        help="Type of attack.",
    )
    parser.add_argument(
        "--output",
        "-O",
        dest="output_dir",
        required=True,
        help="Directory where to store the result.",
    )
    parser.add_argument(
        "--alpha",
        dest="alpha",
        default=0.001,
        type=float,
        help="Step size for the PGD attack or learning rate for the unbounded attack.",
    )
    parser.add_argument(
        "--ball-size",
        dest="ball_size",
        default=0.1,
        type=float,
        help="Linf ball size for the PGD attack.",
    )
    parser.add_argument(
        "--max-iterations",
        "-M",
        dest="iterations",
        default=100,
        type=int,
        help="Maximum number of allowable iterations before terminating.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        default=64,
        type=int,
        help="Batch size for the attack.",
    )
    parser.add_argument("--verbose", "-v", dest="verbose", action="store_true")
    parser.add_argument("--history", "-H", dest="record_history", action="store_true")

    # This to to allow automated testing
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    cudnn.deterministic = True

    # Validate inputs
    if args.origin is not None:
        if args.targeted:
            assert args.target is not None, "Both origin and target must be provided!"
        assert args.n_samples is None, "N cannot be provided when origin and/or target are provided!"
    if args.target is not None:
        assert args.untargeted, "Cannot provide target for untargeted attack!"
        assert args.origin is not None, "Both origin and target must be provided!"
    if args.origin is None and args.target is None:
        assert args.n_samples is not None, "N must be provided when origin and target are not provided!"

    # Get device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} for inference")

    # Load the model and its corresponding transforms
    model, preprocessing_transform, _, _ = get_model(args.model_name, args.model_weights_path)
    model.eval().to(device)

    # Prepare the dataset and the dataloader
    dataset = ImageFolderWithFilenames(root=args.dataset_path, transform=preprocessing_transform)
    if args.origin is not None and args.target is not None:
        origin_index = dataset.get_index_by_filename(args.origin)
        target_index = dataset.get_index_by_filename(args.target)
        dataset = torch.utils.data.Subset(dataset, [origin_index, target_index])
    else:
        if args.targeted:
            dataset = torch.utils.data.Subset(dataset, np.unique(np.linspace(0, len(dataset)-1, 2*args.n_samples, dtype=int)))
        else:
            dataset = torch.utils.data.Subset(dataset, np.unique(np.linspace(0, len(dataset)-1, args.n_samples, dtype=int)))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=(1 if args.untargeted else 2) * args.batch_size,
        shuffle=True,
        num_workers=8,
    )

    # Choose whether we will need noisy initialization of the unbounded attack
    noisy_init = None

    # Timestamp prep
    experiment_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, (batch, names) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):

        with torch.no_grad():
            if args.targeted:
                starting_images, target_images = torch.split(batch, int(len(batch) / 2))
                rep_targets = model(target_images.to(device)).detach()
                starting_names = names[: len(starting_images)]
                target_names = names[len(starting_images) :]
            else:
                starting_images = batch
                rep_targets = None
                starting_names = names
                target_names = [None] * len(names)

        if args.attack_type == "pgd":
            res = pgd_attack(
                model,
                target_representation=rep_targets,
                origin_img=starting_images,
                device=device,
                verbose=args.verbose,
                iterations=args.iterations,
                alpha=args.alpha,
                ball_size=args.ball_size,
                record_history=args.record_history,
            )

        if args.record_history:
            inverted_imgs, histories = res
        else:
            inverted_imgs = res
            histories = [None] * len(names)

        # save the adversarial example (only if a single image)
        if args.origin is not None and args.target is not None:
            path = os.path.join(
                args.output_dir,
                f"{experiment_timestamp}_attacked_{args.target}_from_{args.origin}.png",
            )
            inverted_img = inverse_transform(inverted_imgs.cpu()[0])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            inverted_img.save(path, "png")

        # save attack info which might be useful for further analysis
        for origin_name, target_name, history in zip(starting_names, target_names, histories):
            path = os.path.join(
                args.output_dir,
                f"{experiment_timestamp}_attacked_{origin_name}_from_{target_name}.info",
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)

            info = {
                "origin_name": origin_name,
                "target_name": target_name,
                "model": args.model_name,
                "model_weights_path": args.model_weights_path,
                "dataset": args.dataset_path,
                "attack": args.attack_type,
                "targeted": args.targeted,
                "iterations": args.iterations,
                "alpha": args.alpha,
                "batch_size": args.batch_size,
            }
            if args.record_history:
                info.update(
                    {
                        "dist_l1": history.dist_l1,
                        "dist_l2": history.dist_l2,
                        "dist_linf": history.dist_linf,
                        "perturb_l1": history.perturb_l1,
                        "perturb_l2": history.perturb_l2,
                        "perturb_linf": history.perturb_linf,
                        "cosine_similarity": history.cosine_similarity,
                    }
                )

            if args.attack_type == "pgd":
                info["ball_size"] = args.ball_size

            with open(path, "w") as f:
                yaml.dump(info, f)


if __name__ == "__main__":
    cli()
