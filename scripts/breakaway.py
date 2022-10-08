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
from multiprocessing import Pool
import random

from sklearn.neighbors import NearestNeighbors
from PIL import Image

import argparse

sys.path.append(".")
sys.path.append("..")
sys.path.append("../models")
sys.path.append("models")

from models import get_model

from utils import ImageFolderWithFilenames, pgd_attack


def get_representations_and_labels(model, dataset, device, batch_size, cache_path):
    # compute the representations for the images in the dataset if not done already
    rep_size = model(torch.unsqueeze(dataset.__getitem__(0)[0], 0).to(device)).size()[1]

    cache_loaded_and_valid = False

    if cache_path is not None and os.path.exists(cache_path):
        cache = torch.load(cache_path)
        representations = cache["representations"]
        labels = cache["labels"]
        assert len(representations) == len(dataset) == len(labels), "Sizes of the cache and the dataset do not match!"
        print("Loaded from cache. Checking if valid.")

        # CHECKING IF VALID BY RECOMPUTING THE FIRST FEW REPRESENTATIONS AND COMPARING WITH THE CACHED ONES
        N = 1024
        representations_ = torch.zeros((N, rep_size))

        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, range(N)),
            batch_size=16,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=False,
        )

        t = tqdm.tqdm(total=N, desc="Cache validity check")
        low_idx = 0
        for i, (images, labels_) in enumerate(dataloader):
            representations_[low_idx : low_idx + len(images)] = model(images.to(device)).cpu().detach()
            low_idx += len(images)
            t.update(len(images))
        t.close()

        max_diff = torch.max(torch.abs(representations[:N] - representations_)).item()
        print(f"Max diff: {max_diff}")
        if max_diff < 1e-5:
            cache_loaded_and_valid = True
        print("Cache is VALID." if cache_loaded_and_valid else "Cache is NOT VALID. RECOMPUTING!")

        del representations_
        del dataloader

    if not cache_loaded_and_valid:
        representations = torch.zeros((len(dataset), rep_size))
        labels = list()

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=False,
        )

        t = tqdm.tqdm(total=len(dataset), desc="Cache construction")
        for i, (images, labels_) in enumerate(dataloader):
            representations[i * batch_size : min((i + 1) * batch_size, len(dataset))] = (
                model(images.to(device)).cpu().detach()
            )
            labels += labels_
            t.update(len(images))
        t.close()

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({"representations": representations, "labels": labels}, cache_path)

    return representations, labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Nearest Neighbour evaluation.")
    parser.add_argument(
        "--n-samples",
        dest="n_samples",
        type=int,
        required=True,
        help="Number of images to attack.",
    )
    parser.add_argument(
        "--n-candidates",
        dest="n_candidates",
        type=int,
        default=None,
        help="Number of representations to consider in the NN search.",
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
        dest="output_dir",
        required=True,
        help="Directory where to store the result.",
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
        help="Batch size for the attack and representation computation.",
    )
    parser.add_argument(
        "--cache",
        dest="cache_path",
        default=None,
        type=str,
        help="File to store a cache of the representations so that we don't need to recompute them every time.",
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

    # Prepare the dataset and the dataloaders
    dataset = ImageFolderWithFilenames(root=args.dataset_path, transform=preprocessing_transform)
    print(f"Dataset size: {len(dataset)}")

    representations, labels = get_representations_and_labels(model, dataset, device, args.batch_size, args.cache_path)

    if args.n_candidates is None:
        args.n_candidates = len(dataset)
    representations = representations[: args.n_candidates].clone()
    labels = np.array(labels)[: args.n_candidates]
    print(f"We consider {len(representations)} samples from the dataset as nearest neighbours candidates.")

    # compute attacked representation
    subset_indices = np.unique(np.linspace(0, len(dataset)-1, args.n_samples, dtype=int))
    attack_dataset = torch.utils.data.Subset(dataset, subset_indices)
    attack_dataloader = torch.utils.data.DataLoader(
        attack_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )

    rep_size = model(torch.unsqueeze(dataset.__getitem__(0)[0], 0).to(device)).size()[1]
    
    # keep in GPU if it fits, otherwise keep in RAM and compute distances with the CPU
    if rep_size <= 256:
        representations = representations.to(device)
        rep_on_device = True
        print(f"Moving clean representations to {device}")
    else:
        rep_on_device = False
        print(f"Keeping clean representations in RAM")

        
    # attack_representations = torch.zeros((args.n_samples, rep_size))
    attack_imgs = torch.zeros(*([args.n_samples] + list(dataset.__getitem__(0)[0].size())))
    attack_labels = list()
    
    indices = [] # holds the indices of the top-5 closest clean representations to the adversarial one
    too_close_counts = [] # holds how many of the clean representations are closer to the adversarial one than its own clean one

    t = tqdm.tqdm(total=args.n_samples, desc="Attacks")
    for i, (batch, names) in enumerate(attack_dataloader):
       
        attack_labels += names

        res = pgd_attack(
            model,
            target_representation=None,
            origin_img=batch,
            device=device,
            verbose=False,
            iterations=args.iterations,
            alpha=args.alpha,
            ball_size=args.ball_size,
            record_history=False,
        )
            
        attack_representations = model(res.to(device))
        
        if rep_on_device:
            distances = torch.cdist(attack_representations, representations)
        else:
            distances = torch.cdist(attack_representations.cpu(), representations)
            
        values, ind_ = torch.topk(distances, k=5, largest=False, sorted=True, dim=1)
        indices += ind_.cpu().tolist()
        # assert(len(values)==len(batch))
        # assert(len(ind_)==len(batch))
        # assert(len(ind_[0])==5)
        # assert(values[0][0]==min(values[0]))

        batch_indices = subset_indices[i*args.batch_size:i*args.batch_size+len(batch)]
        attacked_to_clean = torch.sqrt(torch.sum((attack_representations.cpu()-representations[batch_indices].cpu())**2, dim=1))
        too_close_mask = torch.where((distances.cpu() - attacked_to_clean[:, None].cpu()) < 0, 1, 0)
        
        # set the ones corresponding to the correct sample to 0 as 
        # we don't want them to contribute to the risk score
        for i, j in zip(range(len(batch)), batch_indices):
            too_close_mask[i,j] = 0
            
        counts = too_close_mask.sum(dim=1).tolist()
        assert(len(counts) == len(batch))
        too_close_counts += counts
        
        del res
        del attack_representations
        del distances
        del attacked_to_clean
        del too_close_mask
        torch.cuda.synchronize(device)

        t.update(len(batch)) 

    t.close()

    # get the nearest neighbor accuracy
    top_labels = [labels[ns[0]] for ns in indices]
    acc_top1 = sum([true == pred for true, pred in zip(attack_labels, top_labels)]) / args.n_samples
    top5_labels = [labels[ns[:5]] for ns in indices]
    acc_top5 = sum([true in pred for true, pred in zip(attack_labels, top5_labels)]) / args.n_samples
        
    risk = float(np.sum(counts) / (args.n_candidates*args.n_samples))
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Acc top 1: {acc_top1*100:.2f} %")
    print(f"Acc top 5: {acc_top5*100:.2f} %")
    print(f"Empirical risk of unsupervised attack being closer to another sample: {risk*100} %")

    info = {
        "model": args.model_name,
        "model_weights_path": args.model_weights_path,
        "dataset": args.dataset_path,
        "iterations": args.iterations,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "ball_size": args.ball_size,
        "n_samples": args.n_samples,
        "n_candidates": args.n_candidates,
        "acc_top1": acc_top1,
        "acc_top5": acc_top5,
        "risk": risk,
    }

    experiment_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"{experiment_timestamp}_nn.info"), "w") as f:
        yaml.dump(info, f)
