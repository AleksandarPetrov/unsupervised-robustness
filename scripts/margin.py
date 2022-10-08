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
import pandas as pd
import random

from sklearn.neighbors import NearestNeighbors

import argparse

sys.path.append(".")
sys.path.append("..")
sys.path.append("../models")
sys.path.append("models")

from models import get_model
from utils import ImageFolderWithFilenames, pgd_attack


parser = argparse.ArgumentParser(description="Unsupervised Average Neighbourhood Margin evaluation.")
parser.add_argument(
    "--n-samples",
    dest="n_samples",
    type=int,
    required=True,
    help="Number of pairs to attack.",
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
parser.add_argument(
    "--output",
    "-O",
    dest="output",
    required=True,
    help="File where to store the result.",
)
parser.add_argument(
    "--alpha",
    dest="alpha",
    default=0.001,
    type=float,
    help="Step size for the attack.",
)
parser.add_argument(
    "--ball-size",
    dest="ball_size",
    default=0.05,
    type=float,
    help="Linf ball size for the PGD attack.",
)
parser.add_argument(
    "--iterations",
    "-M",
    dest="iterations",
    default=3,
    type=int,
    help="Maximum number of allowable iterations of the attack before terminating.",
)
parser.add_argument(
    "--batch-size",
    "-b",
    dest="batch_size",
    default=64,
    type=int,
    help="Batch size for the attack computation.",
)

args = parser.parse_args()
print(args)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
cudnn.deterministic = True

# Get device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} for inference")

# Load the model and its corresponding transforms
model, preprocessing_transform, inverse_transform, training_augmentation = get_model(
    args.model_name, args.model_weights_path
)
model.eval().to(device)

# Prepare the dataset
dataset = ImageFolderWithFilenames(root=args.dataset_path, transform=preprocessing_transform)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2*args.batch_size,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    drop_last=False,
)

# compute the representations for the images in the dataset
rep_size = model(torch.unsqueeze(dataset.__getitem__(0)[0], 0).to(device)).size()[1]

d_x_xtoxp, d_x_xptox, d_xp_xtoxp, d_xp_xptox, d_x_xp = [], [], [], [], []

tb = tqdm.tqdm(total=args.n_samples, desc="Attacks")

while True:
    for batch, _ in dataloader:
        x, xp = batch.split(args.batch_size)
        
        x=x.to(device)
        xp=xp.to(device)
        
        with torch.no_grad():
            rep_x = model(x).detach()
            rep_xp = model(xp).detach()
            
        x_to_xp = pgd_attack(
            model,
            target_representation=rep_xp,
            origin_img=x,
            device=device,
            verbose=False,
            iterations=args.iterations,
            alpha=args.alpha,
            ball_size=args.ball_size,
            record_history=False,
        )
        xp_to_x = pgd_attack(
            model,
            target_representation=rep_x,
            origin_img=xp,
            device=device,
            verbose=False,
            iterations=args.iterations,
            alpha=args.alpha,
            ball_size=args.ball_size,
            record_history=False,
        )
            
        with torch.no_grad():
            rep_x_to_xp = model(x_to_xp.to(device)).detach()
            rep_xp_to_x = model(xp_to_x.to(device)).detach()
            
        d_x_xp += torch.sqrt(torch.sum((rep_x-rep_xp)**2, dim=1)).tolist()
        d_x_xtoxp += torch.sqrt(torch.sum((rep_x-rep_x_to_xp)**2, dim=1)).tolist()
        d_x_xptox += torch.sqrt(torch.sum((rep_x-rep_xp_to_x)**2, dim=1)).tolist()
        d_xp_xtoxp += torch.sqrt(torch.sum((rep_xp-rep_x_to_xp)**2, dim=1)).tolist()
        d_xp_xptox += torch.sqrt(torch.sum((rep_xp-rep_xp_to_x)**2, dim=1)).tolist()
    
        tb.update(len(x))
    
        if len(d_x_xp) >= args.n_samples:
            break
    
    if len(d_x_xp) >= args.n_samples:
        break
        
os.makedirs(os.path.dirname(args.output), exist_ok=True)
df = pd.DataFrame({
    "d_x_xp": d_x_xp,
    "d_x_xtoxp": d_x_xtoxp,
    "d_x_xptox": d_x_xptox,
    "d_xp_xtoxp": d_xp_xtoxp,
    "d_xp_xptox": d_xp_xptox,    
})
df.to_csv(args.output, index=False)