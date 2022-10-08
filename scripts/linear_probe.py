#!/usr/bin/env python
# Inspired by https://github.com/facebookresearch/moco/blob/main/main_lincls.py
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import yaml
import sys
import numpy as np
from PIL import Image

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

sys.path.append(".")
sys.path.append("..")
sys.path.append("../models")
sys.path.append("models")
sys.path.append("../moco")
sys.path.append("moco")

from models import get_model, NativeScalerWithGradNormCount
from moco.loader import GaussianNoise

parser = argparse.ArgumentParser(description="Linear Probe on ImageNet")
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
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument("--epochs", default=25, type=int, metavar="N", help="number of total epochs to run")
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
    default=30.0,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[15, 20],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by a ratio)",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0.0,
    type=float,
    metavar="W",
    help="weight decay (default: 0.)",
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
    "-e",
    "--evaluate",
    dest="evaluate",
    default=None,
    help="evaluate model on validation set and save to specified file",
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
parser.add_argument("--checkpoint-dir", default="", type=str, help="Directory to store checkpoints.")
parser.add_argument(
    "--low_pass_radius",
    default=None,
    type=int,
    help="Radius for the low-pass filter, if not provided no filtering.",
)
parser.add_argument(
    "--noise-sigma",
    default=None,
    type=float,
    help="Sigma for the Gaussian noise, if not provided no noise is added (should be between 0 and 1).",
)

best_acc1 = 0


class IdentityTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class LowPassFilter:
    """
    A Low-Pass Filter as in https://arxiv.org/abs/1905.13545
    """

    def __init__(self, radius) -> None:
        self.radius = radius
        self.mask = None

    def make_mask(self, img):
        def thresholding(i, j, imageSize, r):
            dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
            return 1.0 if dis < r else 0.0

        rows, cols, _ = img.shape
        mask = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                mask[i, j] = thresholding(i, j, imageSize=rows, r=self.radius)
        self.mask = mask

    def __call__(self, img):
        """
        Args:
            pic (PIL Image): Image to have its high-frequency components removed.
        Returns:
            PIL Image: Filtered image.
        """

        img = np.array(img)

        if self.mask is None:
            self.make_mask(img)

        img_low = np.zeros_like(img)
        for j in range(3):
            fd = np.fft.fftshift(np.fft.fft2(img[:, :, j]))
            fd = fd * self.mask
            img_low[:, :, j] = np.clip(np.abs(np.fft.ifft2(np.fft.ifftshift(fd))), 0, 255)

        return Image.fromarray(img_low)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


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

    # if resume is not provided but a checkpoint is found in the checkpoint directory, resume from there
    if not args.resume and os.path.exists(args.checkpoint_dir):
        possible_checkpoints = os.listdir(args.checkpoint_dir)
        possible_checkpoints = [f for f in possible_checkpoints if f.startswith("checkpoint_") and f.endswith(".pth.tar")]
        possible_checkpoints.sort()
        if len(possible_checkpoints)>0:
            args.resume = os.path.join(args.checkpoint_dir, possible_checkpoints[-1])
        print(f"Found {len(possible_checkpoints)} checkpoints in {args.checkpoint_dir}:")
        for f in possible_checkpoints:
            print(f"- {f}")
        print(f"RESTORING TRAINING FROM {args.resume}")
 
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

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
    global best_acc1
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
    print("=> creating model '{}' with weights from {}".format(args.model_name, args.model_weights_path))

    model, preprocessing_transform, inverse_transform, _ = get_model(
        args.model_name,
        args.model_weights_path,
        for_lin_probe=True,
        resnet_fc_init=(args.model_name == "resnet" and args.evaluate is None),
    )
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
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    # parameters = list(model.module.linear_layer.parameters())
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

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
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.dataset_path, "train")
    valdir = os.path.join(args.dataset_path, "val")

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                LowPassFilter(radius=args.low_pass_radius) if args.low_pass_radius is not None else IdentityTransform(),
                GaussianNoise(std=255 * args.noise_sigma) if args.noise_sigma is not None else IdentityTransform(),
                preprocessing_transform,
            ]
        ),
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
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    LowPassFilter(radius=args.low_pass_radius)
                    if args.low_pass_radius is not None else IdentityTransform(),
                    GaussianNoise(std=255 * args.noise_sigma) if args.noise_sigma is not None else IdentityTransform(),
                    preprocessing_transform,
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    ### Save samples once just for sanity check
    if (args.gpu == 0 or args.gpu is None) and args.evaluate is not None:
        os.makedirs(os.path.dirname(args.evaluate), exist_ok=True)
        im_train = inverse_transform(torchvision.utils.make_grid(next(iter(train_loader))[0]))
        im_train.save(os.path.join(os.path.dirname(args.evaluate), f"training_set_sample.png"))
        im_val = inverse_transform(torchvision.utils.make_grid(next(iter(val_loader))[0]))
        im_val.save(os.path.join(os.path.dirname(args.evaluate), f"validation_set_sample.png"))

    if args.evaluate is not None:
        acc1, acc5 = validate(val_loader, model, criterion, args)

        os.makedirs(os.path.dirname(args.evaluate), exist_ok=True)
        with open(args.evaluate, "w") as f:
            yaml.dump({"Top-1 accuracy": float(acc1), "Top-5 accuracy": float(acc5)}, f)

        return

    if args.model_name == "mae":
        loss_scaler = NativeScalerWithGradNormCount()
    else:
        loss_scaler = None

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            args,
            loss_scaler=loss_scaler,
        )

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                filename="checkpoint_{:04d}.pth.tar".format(epoch),
                checkpoint_dir=args.checkpoint_dir,
                args=args,
            )
            # if epoch == args.start_epoch:
            #     sanity_check(model.state_dict(), args.pretrained)


def train(train_loader, model, criterion, optimizer, epoch, args, loss_scaler):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        if args.model_name.startswith("simclr2_"):
            images = images.requires_grad_()
            output = model(images, apply_fc=True)
        else:
            output = model(images)

        loss = criterion(output, target)

        if args.model_name == "mae":
            loss_scaler(
                loss,
                optimizer,
                clip_grad=None,
                parameters=model.parameters(),
                create_graph=False,
                update_grad=True,
            )

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # the MAE scaler handles the backprop and optimization,
        # for the other models we do it here
        if args.model_name != "mae":
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.model_name.startswith("simclr2_"):
                output = model(images, apply_fc=True)
            else:
                output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))

    return top1.avg, top5.avg


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
        shutil.copyfile(
            os.path.join(checkpoint_dir, filename),
            os.path.join(checkpoint_dir, "model_best.pth.tar"),
        )


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint["state_dict"]

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if "fc.weight" in k or "fc.bias" in k:
            continue

        # name in pretrained model
        k_pre = "module.encoder_q." + k[len("module.") :] if k.startswith("module.") else "module.encoder_q." + k

        assert (
            state_dict[k].cpu() == state_dict_pre[k_pre]
        ).all(), "{} is changed in linear classifier training.".format(k)

    print("=> sanity check passed.")


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
