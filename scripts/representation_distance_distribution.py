#!/usr/bin/env python3

import sys
import os
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import random
import tqdm
import numpy as np

import argparse

sys.path.append(".")
sys.path.append("..")
sys.path.append("../models")
sys.path.append("models")

from models import get_model

from utils.tools import ImageFolderWithFilenames

parser = argparse.ArgumentParser(description="Computes the distribution of inter-representational distances.")
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
parser.add_argument(
    "--n-samples",
    "-N",
    dest="n_samples",
    type=int,
    default=None,
    help="Number of samples to take from the dataset.",
)
parser.add_argument(
    "--batch-size",
    "-b",
    dest="batch_size",
    type=int,
    default=256,
    help="Batch size, all pairwise distances in the batch will be considered.",
)
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "-o",
    "--output",
    default=None,
    dest="output",
    type=str,
    help="where to store the resulting quantiles",
)

args = parser.parse_args()

if args.output is None:
    args.output = args.model_weights_path + ".quantiles"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} for inference")

model, preprocessing_transform, _, _ = get_model(args.model_name, args.model_weights_path)
model.eval().to(device)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
cudnn.deterministic = True

dataset = ImageFolderWithFilenames(root=args.dataset_path, transform=preprocessing_transform)
if args.n_samples is not None:
    dataset = torch.utils.data.Subset(dataset, np.unique(np.linspace(0, len(dataset)-1, args.n_samples, dtype=int)))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

l2_distances, l1_distances, linf_distances = [], [], []

for idx, (batch, _) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
    with torch.no_grad():
        representations = model(batch.to(device))
        l1_distances += F.pdist(representations, p=1).tolist()
        l2_distances += F.pdist(representations, p=2).tolist()
        linf_distances += F.pdist(representations, p=float("inf")).tolist()

quantile_values = list(np.arange(0.001, 1.0, 0.001).astype(float)) + list(np.logspace(-9, -3, 100, endpoint=False))
quantile_values.sort()
quantiles_l1 = np.quantile(l1_distances, quantile_values)
quantiles_l2 = np.quantile(l2_distances, quantile_values)
quantiles_linf = np.quantile(linf_distances, quantile_values)

quantiles = dict()
quantiles["l1"] = {float(k): float(v) for k, v in zip(quantile_values, quantiles_l1)}
quantiles["l2"] = {float(k): float(v) for k, v in zip(quantile_values, quantiles_l2)}
quantiles["linf"] = {float(k): float(v) for k, v in zip(quantile_values, quantiles_linf)}

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    yaml.dump(quantiles, f)
