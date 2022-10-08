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
sys.path.append("../smoothing/code")
sys.path.append("smoothing/code")
from core import Smooth
from models import get_model


class ClippedSmooth(Smooth):
    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for k in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device="cuda") * self.sigma

                # We clip to be within the valid model input.
                # This does not affect our results as this step
                # can be considered part of the model.
                noisy = torch.clip(batch + noise, 0, 1)

                predictions = self.base_classifier(noisy).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Computing certified accuracy with randomized smoothing")
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
        "--checkpoint",
        required=True,
        type=str,
        metavar="PATH",
        help="path to linprobe checkpoint which we evaluate",
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
    parser.add_argument("--N0", type=int, default=100)
    parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
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

    model_linprobe, preprocessing_transform, inverse_transform, _ = get_model(
        args.model_name,
        args.model_weights_path,
        for_lin_probe=True,
        resnet_fc_init=False,
    )

    model_linprobe = torch.nn.DataParallel(model_linprobe).cuda()
    model_linprobe.eval().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_linprobe.load_state_dict(checkpoint["state_dict"])

    # Bake the final layer application into the model for simclr2:
    if args.model_name.startswith("simclr2_"):
        model_linprobe.__old_forward = model_linprobe.forward
        model_linprobe.forward = lambda x: model_linprobe.__old_forward(x, apply_fc=True)

    # prepare output file
    f = open(args.output, "w")
    print("idx,\tlabel,\tpredict,\tradius,\tcorrect,\tclean_correct,\ttime", file=f, flush=True)

    dataset = torchvision.datasets.ImageFolder(root=args.dataset, transform=preprocessing_transform)

    smoothed_classifier = ClippedSmooth(model_linprobe, 1000, args.sigma)

    indices = np.unique(np.linspace(0, len(dataset)-1, args.n_samples, dtype=int))

    for i in tqdm.tqdm(indices):

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()

        # check if the clean sample is correctly classified
        clean_predictions = smoothed_classifier.base_classifier(x.repeat((1, 1, 1, 1))).argmax(1)[0]
        clean_correct = int(clean_predictions == label)

        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch_size)

        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print(
            "{},\t{},\t{},\t{:.3},\t{},\t{},\t{}".format(
                i, label, prediction, radius, correct, clean_correct, time_elapsed
            ),
            file=f,
            flush=True,
        )

    f.close()
