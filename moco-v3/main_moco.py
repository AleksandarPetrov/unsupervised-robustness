#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import moco.builder
import moco.loader
import moco.optimizer

import vits

import yaml
import sys

sys.path.append("..")
from utils import pgd_attack

torchvision_model_names = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__") and callable(torchvision_models.__dict__[name])
)

model_names = ["vit_small", "vit_base", "vit_conv_small", "vit_conv_base"] + torchvision_model_names

parser = argparse.ArgumentParser(description="MoCo ImageNet Pre-Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j", "--workers", default=32, type=int, metavar="N", help="number of data loading workers (default: 32)"
)
parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument(
    "-b",
    "--batch-size",
    default=4096,
    type=int,
    metavar="N",
    help="mini-batch size (default: 4096), this is the total "
    "batch size of all GPUs on all nodes when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr", "--learning-rate", default=0.6, type=float, metavar="LR", help="initial (base) learning rate", dest="lr"
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-6,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-6)",
    dest="weight_decay",
)
parser.add_argument("-p", "--print-freq", default=10, type=int, metavar="N", help="print frequency (default: 10)")
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument("--world-size", default=-1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# moco specific configs:
parser.add_argument("--moco-dim", default=256, type=int, help="feature dimension (default: 256)")
parser.add_argument("--moco-mlp-dim", default=4096, type=int, help="hidden dimension in MLPs (default: 4096)")
parser.add_argument(
    "--moco-m", default=0.99, type=float, help="moco momentum of updating momentum encoder (default: 0.99)"
)
parser.add_argument(
    "--moco-m-cos",
    action="store_true",
    help="gradually increase moco momentum to 1 with a " "half-cycle cosine schedule",
)
parser.add_argument("--moco-t", default=1.0, type=float, help="softmax temperature (default: 1.0)")

# vit specific configs:
parser.add_argument("--stop-grad-conv1", action="store_true", help="stop-grad after first conv, or patch embedding")

# other upgrades
parser.add_argument(
    "--optimizer", default="lars", type=str, choices=["lars", "adamw"], help="optimizer used (default: lars)"
)
parser.add_argument("--warmup-epochs", default=10, type=int, metavar="N", help="number of warmup epochs")
parser.add_argument("--crop-min", default=0.08, type=float, help="minimum scale for random cropping (default: 0.08)")

# Unsupervised robustness options:
parser.add_argument("--checkpoint-dir", default="", type=str, help="Directory to store checkpoints.")
parser.add_argument(
    "--attack",
    default=None,
    choices=[
        "pgd-untargeted",
        "pgd-targeted",
        "l-pgd",
    ],
    help="Attack for adversarial training.",
)
parser.add_argument(
    "--ball-size",
    default=0.05,
    type=float,
    help="Radius of the Linf ball for attacks requiring it.",
)
parser.add_argument("--attack-iterations", default=3, type=int, help="Number of iterations for attack.")
parser.add_argument(
    "--attack-alpha",
    default=0.001,
    type=float,
    help="Optimization step size for attack.",
)


def main():
    args = parser.parse_args()

    if args.dist_url is None:
        import socket

        free_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        free_socket.bind(("0.0.0.0", 0))
        free_socket.listen(5)
        port = free_socket.getsockname()[1]
        free_socket.close()
        args.dist_url = f"tcp://localhost:{port}"

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn("You have chosen a specific GPU. This will completely " "disable data parallelism.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # if resume is not provided but a checkpoint is found in the checkpoint directory, resume from there
    if not args.resume and os.path.exists(args.checkpoint_dir):
        possible_checkpoints = os.listdir(args.checkpoint_dir)
        possible_checkpoints = [
            f for f in possible_checkpoints if f.startswith("checkpoint_") and f.endswith(".pth.tar")
        ]
        possible_checkpoints.sort()
        if len(possible_checkpoints) > 0:
            args.resume = os.path.join(args.checkpoint_dir, possible_checkpoints[-1])
        print(f"Found {len(possible_checkpoints)} checkpoints in {args.checkpoint_dir}:")
        for f in possible_checkpoints:
            print(f"- {f}")
        print(f"RESTORING TRAINING FROM {args.resume}")

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith("vit"):
        model = moco.builder.MoCo_ViT(
            partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
            args.moco_dim,
            args.moco_mlp_dim,
            args.moco_t,
        )
    else:
        model = moco.builder.MoCo_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True),
            args.moco_dim,
            args.moco_mlp_dim,
            args.moco_t,
        )

    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256

    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model)  # print model after SyncBatchNorm

    if args.optimizer == "lars":
        optimizer = moco.optimizer.LARS(
            model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler()
    summary_writer = SummaryWriter() if args.rank == 0 else None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if args.start_epoch == 0:
                args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])

            if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                print("=> OPTIMIZER NOT FOUND IN CHECKPOINT, WILL START FROM CLEAN SLATE!")

            if "scaler" in checkpoint and checkpoint["scaler"] is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            else:
                print("=> SCALER NOT FOUND IN CHECKPOINT, WILL START FROM CLEAN SLATE!")

            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.attack is not None:
        print(
            f"=> ATTACK {args.attack} will be used with alpha={args.attack_alpha}"
            + (f", ball size={args.ball_size}" if args.attack is not None and args.attack.startswith("pgd") else "")
            + f" for {args.attack_iterations} iterations."
        )

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, "train")
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),  # not strengthened
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),  # not strengthened
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.1),
        transforms.RandomApply([moco.loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize
    ]

    train_dataset = datasets.ImageFolder(
        traindir, moco.loader.TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2))
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank == 0
        ):  # only the first GPU saves checkpoint
            save_checkpoint(
                state={
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                is_best=False,
                filename="checkpoint_%04d.pth.tar" % epoch,
                checkpoint_dir=args.checkpoint_dir,
                args=args,
            )

    if args.rank == 0:
        summary_writer.close()


def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    learning_rates = AverageMeter("LR", ":.4e")
    losses = AverageMeter("Loss", ":.4e")
    attack_time = AverageMeter("Attack", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, attack_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m

    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        attack_start = time.time()

        if args.attack is None:
            pass
        # if targeted we will need representations for the target,
        # we pick the next image in the batch as the target
        elif args.attack.endswith("-targeted"):
            with torch.no_grad(), torch.cuda.amp.autocast(True):
                target_representation = torch.roll(model(images[1]), shifts=1, dims=0).detach()
        else:
            target_representation = None

        if args.attack is None:
            pass
        elif args.attack.startswith("pgd"):
            attacked_images = pgd_attack(
                model=model,
                target_representation=target_representation,
                origin_img=images[0],
                device=f"cuda:{args.gpu}",
                verbose=False,
                iterations=args.attack_iterations,
                ball_size=args.ball_size,
                alpha=args.attack_alpha,
                record_history=False,
                normalization_fn=normalize,
                amp=True,
            )

        elif args.attack == "l-pgd":
            perturbation = (
                torch.empty_like(images[0]).uniform_(-args.ball_size, args.ball_size).cuda(args.gpu, non_blocking=True)
            )
            perturbation.requires_grad = True

            for _ in range(args.attack_iterations):

                with torch.cuda.amp.autocast(True):
                    loss_ = model(x1=normalize(images[0] + perturbation), x2=normalize(images[1]), m=moco_m)

                loss_.backward()
                model.zero_grad()

                with torch.no_grad():
                    sign_data_grad = perturbation.grad.sign()
                    perturbation.add_(args.attack_alpha * sign_data_grad)
                    perturbation.clamp_(min=-args.ball_size, max=args.ball_size)
                    perturbation.grad.data.zero_()
                    torch.maximum(perturbation, 0 - images[0], out=perturbation)
                    torch.minimum(perturbation, 1 - images[0], out=perturbation)

            with torch.no_grad():
                attacked_images = images[0] + perturbation

        else:
            raise RuntimeError(f"Attack {args.attack} not recognized!")

        attack_time.update(time.time() - attack_start)

        # Apply normalization
        images[0], images[1] = normalize(images[0]), normalize(images[1])

        if args.attack is not None:
            attacked_images = normalize(attacked_images)

        ### Save samples each epoch just for sanity check
        if args.gpu == 0 and i == 0:
            inverse_transform = transforms.Compose(
                [
                    transforms.transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                    transforms.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
                    transforms.ToPILImage(),
                ]
            )
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            im0 = inverse_transform(make_grid(images[0]))
            im0.save(os.path.join(args.checkpoint_dir, f"training_set_sample_epoch{epoch}_0.png"))
            im1 = inverse_transform(make_grid(images[1]))
            im1.save(os.path.join(args.checkpoint_dir, f"training_set_sample_epoch{epoch}_1.png"))

            if args.attack is not None:
                im_attacked = inverse_transform(make_grid(attacked_images))
                im_attacked.save(os.path.join(args.checkpoint_dir, f"training_set_sample_epoch{epoch}_attacked.png"))

        # compute output
        with torch.cuda.amp.autocast(True):
            if args.attack is not None:
                loss = model(images[0], images[1], moco_m, attacked_x1=attacked_images)
            else:
                loss = model(images[0], images[1], moco_m)

        losses.update(loss.item(), images[0].size(0))
        if args.rank == 0:
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", checkpoint_dir="", args=None):
    os.makedirs(
        os.path.dirname(os.path.abspath(os.path.join(checkpoint_dir, filename))),
        exist_ok=True,
    )
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if args is not None:
        with open(os.path.join(checkpoint_dir, filename + ".args"), "w") as f:
            yaml.dump(vars(args), f)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, "model_best.pth.tar"))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = (
            args.lr
            * 0.5
            * (1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1.0 - 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs)) * (1.0 - args.moco_m)
    return m


if __name__ == "__main__":
    main()
