#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Parts of this code have been modified by the authors of the repository
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import yaml

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import moco.loader
import moco.builder

import sys

torch.autograd.set_detect_anomaly(True)

sys.path.append("..")
from utils import pgd_attack

model_names = sorted(
    name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
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
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
parser.add_argument(
    "--dist-url",
    default=None,
    type=str,
    help="url used to set up distributed training",
)
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
parser.add_argument("--moco-dim", default=128, type=int, help="feature dimension (default: 128)")
parser.add_argument(
    "--moco-k",
    default=65536,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument("--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)")

# options for moco v2
parser.add_argument("--mlp", action="store_true", help="use mlp head")
parser.add_argument("--aug-plus", action="store_true", help="use moco v2 data augmentation")
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

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

    # # if resume is not provided but a checkpoint is found in the checkpoint directory, resume from there
    # if not args.resume and os.path.exists(args.checkpoint_dir):
    #     possible_checkpoints = os.listdir(args.checkpoint_dir)
    #     possible_checkpoints = [f for f in possible_checkpoints if f.startswith("checkpoint_") and f.endswith(".pth.tar")]
    #     possible_checkpoints.sort()
    #     if len(possible_checkpoints)>0:
    #         args.resume = os.path.join(args.checkpoint_dir, possible_checkpoints[-1])
    #     print(f"Found {len(possible_checkpoints)} checkpoints in {args.checkpoint_dir}:")
    #     for f in possible_checkpoints:
    #         print(f"- {f}")
    #     print(f"RESTORING TRAINING FROM {args.resume}")
 
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

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

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
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim,
        args.moco_k,
        args.moco_m,
        args.moco_t,
        args.mlp,
    )
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

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
            #             print(checkpoint['state_dict'].keys())

            # the k encoder weights are missing from the released pretrained model
            # so we initialize the k encoder with the weights for the q encoder
            new_parameters = {}
            for q_parameter in checkpoint["state_dict"].keys():
                if q_parameter.startswith("module.encoder_q."):
                    k_parameter = f"module.encoder_k.{q_parameter[len('module.encoder_q.'):]}"
                    if k_parameter not in checkpoint["state_dict"].keys():
                        new_parameters[k_parameter] = checkpoint["state_dict"][q_parameter]

            if len(new_parameters) > 0:
                print("=> COPYING THE q-ENCODER WEIGHTS TO THE k-ENCODER")
                checkpoint["state_dict"].update(new_parameters)

            model.load_state_dict(checkpoint["state_dict"], strict=False)

            args.start_epoch = checkpoint["epoch"]

            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                for e in range(args.start_epoch):
                    adjust_learning_rate(optimizer, e, args)
                print("=> OPTIMIZER NOT FOUND IN CHECKPOINT, SIMULATED PAST EPOCHS")

            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.attack is not None:
        print(
            f"=> ATTACK {args.attack} will be used with alpha={args.attack_alpha}"
            + (
                f", ball size={args.ball_size}"
                if args.attack is not None and (args.attack.startswith("pgd") or args.attack == "rocl")
                else ""
            )
            + f" for {args.attack_iterations} iterations."
        )

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, "train")
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]

    train_dataset = datasets.ImageFolder(traindir, moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

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
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename="checkpoint_{:04d}.pth.tar".format(epoch),
                checkpoint_dir=args.checkpoint_dir,
                args=args,
            )


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    attack_time = AverageMeter("Attack", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, attack_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    model.train()
    
    second_criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        attack_start = time.time()

        if args.attack is None:
            pass
        # if targeted we will need representations for the target,
        # we pick the next image in the batch as the target
        elif args.attack.endswith("-targeted"):
            with torch.no_grad():
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
            )

        elif args.attack == "l-pgd":
            perturbation = (
                torch.empty_like(images[0]).uniform_(-args.ball_size, args.ball_size).cuda(args.gpu, non_blocking=True)
            )
            perturbation.requires_grad = True

            for _ in range(args.attack_iterations):
                
                output_, target_ = model(im_q=normalize(images[0] + perturbation), im_k=normalize(images[1]))
                loss_ = criterion(output_, target_)
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
            inverse_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                    torchvision.transforms.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
                    torchvision.transforms.ToPILImage(),
                ]
            )
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            im0 = inverse_transform(torchvision.utils.make_grid(images[0]))
            im0.save(os.path.join(args.checkpoint_dir, f"training_set_sample_epoch{epoch}_0.png"))
            im1 = inverse_transform(torchvision.utils.make_grid(images[1]))
            im1.save(os.path.join(args.checkpoint_dir, f"training_set_sample_epoch{epoch}_1.png"))
            
            if args.attack is not None:
                im_attacked = inverse_transform(torchvision.utils.make_grid(attacked_images))
                im_attacked.save(os.path.join(args.checkpoint_dir, f"training_set_sample_epoch{epoch}_attacked.png"))
                
        
        output, target = model(im_q=images[0], im_k=images[1], im_attacked=attacked_images)
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output[:len(images[0])], target[:len(images[0])], topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
