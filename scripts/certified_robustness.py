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
from time import time


from math import ceil
from scipy.stats import norm, binom_test

import argparse

sys.path.append(".")
sys.path.append("..")
sys.path.append("../models")
sys.path.append("models")

from utils import CenterSmooth
from models import get_model


def l2_dist(batch1, batch2):
    dist = torch.norm(torch.flatten(batch1 - batch2, start_dim=1), p=2, dim=1)
    dist = dist.cpu().numpy()
    return dist


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Computing certified robustness with center smoothing")
    parser.add_argument(
        "--model",
        "-m",
        dest="model_name",
        required=True,
        help="Which model we use.",
    )
    parser.add_argument(
        "--dataset",
        "-D",
        dest="dataset",
        required=True,
        help="Directory containing the dataset to use.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size (default: 128), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--model-weights",
        dest="model_weights_path",
        required=True,
        help="Path to the model weights.",
    )
    parser.add_argument(
        "--sigma",
        default=0.25,
        type=float,
        help="Sigma for the Gaussian noise.",
    )
    parser.add_argument(
        "--n-samples",
        dest="n_samples",
        type=int,
        default=None,
        help="Number of samples to take from the dataset.",
    )
    parser.add_argument("--N0", type=int, default=10_000)
    parser.add_argument("--N", type=int, default=1_000_000, help="number of samples to use")
    parser.add_argument("--output", type=str, required=True, help="file to save the results to")

    args = parser.parse_args()
    print(args)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    cudnn.deterministic = True

    # Get device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} for inference")

    model, preprocessing_transform, inverse_transform, _ = get_model(
        args.model_name,
        args.model_weights_path,
        for_lin_probe=False,
    )

    model.eval().to(device)

    # prepare output file
    f = open(args.output, "w")
    print("idx,\teps_out,\tsmoothing_error,\ttime", file=f, flush=True)

    dataset = torchvision.datasets.ImageFolder(root=args.dataset, transform=preprocessing_transform)

    smoothed_classifier = CenterSmooth(model, l2_dist, args.sigma, n_pred=args.N0, n_cert=args.N)

    indices = np.unique(np.linspace(0, len(dataset)-1, args.n_samples, dtype=int))

    for i in tqdm.tqdm(indices):

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()

        # check if the clean sample is correctly classified

        eps_out, smoothing_error = smoothed_classifier.certify(
            x, args.sigma, batch_size=args.batch_size
        )  # implicitly setting h=1

        after_time = time()

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print(
            "{},\t{:.3},\t{:.3},\t{}".format(i, float(eps_out), float(smoothing_error), time_elapsed),
            file=f,
            flush=True,
        )

    f.close()
