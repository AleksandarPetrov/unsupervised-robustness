#!/usr/bin/env python3

import sys
import os
import torch
import torchvision
import torch.backends.cudnn as cudnn
import numpy as np
import yaml
import random
from PIL import Image, ImageFont, ImageDraw
import tqdm

sys.path.append(".")
sys.path.append("..")
sys.path.append("../models")
sys.path.append("models")

from models import get_model
from utils import pgd_attack

import argparse

IMAGENET_DOGS_INDICES = list(range(151, 275))
IMAGENET_CATS_INDICES = list(range(281, 294))


parser = argparse.ArgumentParser(description="Dogs and cats impersonation evaluation")
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
    help="Directory containing the PetImages dataset.",
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
    "--iterations",
    "-M",
    dest="iterations",
    default=100,
    type=int,
    help="Maximum number of allowable iterations before terminating.",
)

parser.add_argument("--output", type=str, required=True, help="file to save the results to (w/o extension)")

args = parser.parse_args()

assert args.iterations % 10 == 0, "Iterations should be a multiple of 10 in order to prevent surprising effects"

print(args)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
cudnn.deterministic = True

results = dict()

# Get device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} for inference")

# prepare the models
model_rep, _, _, _ = get_model(args.model_name, args.model_weights_path, for_lin_probe=False)
model_rep.eval().to(device)

model_linprobe, preprocessing_transform, inverse_transform, _ = get_model(
    args.model_name, args.model_weights_path, for_lin_probe=True
)
model_linprobe = torch.nn.DataParallel(model_linprobe).cuda()
model_linprobe.eval().to(device)
checkpoint = torch.load(args.checkpoint, map_location=device)
model_linprobe.load_state_dict(checkpoint["state_dict"])

# prepare the datasets
dataset = torchvision.datasets.ImageFolder(root=args.dataset, transform=preprocessing_transform)
dl = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=False)

# filter only the correctly classified ones
detected_as_cats = []
detected_as_dogs = []
for idx, (batch, _) in tqdm.tqdm(enumerate(dl), total=len(dl), desc="Extracting the correctly classified images"):

    if args.model_name.startswith("simclr2_"):
        predictions = torch.argmax(model_linprobe(batch, apply_fc=True), axis=1)
    else:
        predictions = torch.argmax(model_linprobe(batch), axis=1)

    detected_as_dogs += [(pred in IMAGENET_DOGS_INDICES) for pred in predictions]
    detected_as_cats += [(pred in IMAGENET_CATS_INDICES) for pred in predictions]

correct_cats = [idx for idx, gt in enumerate(dataset.targets) if (gt == 0 and detected_as_cats[idx])]
correct_dogs = [idx for idx, gt in enumerate(dataset.targets) if (gt == 1 and detected_as_dogs[idx])]

results["Cat detection accuracy"] = float(len(correct_cats) / len(np.where(np.array(dataset.targets) == 0)[0]))
results["Dog detection accuracy"] = float(len(correct_dogs) / len(np.where(np.array(dataset.targets) == 1)[0]))

correct_dogs_dataset = torch.utils.data.Subset(dataset, correct_dogs)
correct_cats_dataset = torch.utils.data.Subset(dataset, correct_cats)
dogs_dataloader = torch.utils.data.DataLoader(
    correct_dogs_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True
)
cats_dataloader = torch.utils.data.DataLoader(
    correct_cats_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True
)

correct_cats_sample = inverse_transform(
    torchvision.utils.make_grid([correct_cats_dataset.__getitem__(i)[0] for i in range(32)])
).resize((800, 400))
correct_dogs_sample = inverse_transform(
    torchvision.utils.make_grid([correct_dogs_dataset.__getitem__(i)[0] for i in range(32)])
).resize((800, 400))

del batch
torch.cuda.empty_cache()
torch.cuda.synchronize()

attack_super_steps = list(range(0, args.iterations + 1, 10)) + [3, 5]
attack_super_steps.pop(0)
attack_super_steps.sort()

successful_cat_impersonators = {step: [] for step in attack_super_steps}
successful_dog_impersonators = {step: [] for step in attack_super_steps}

for idx, ((cat_batch, _), (dog_batch, _)) in tqdm.tqdm(
    enumerate(zip(cats_dataloader, dogs_dataloader)),
    total=min(len(cats_dataloader), len(dogs_dataloader)),
    desc="Impersonation attacks",
):

    with torch.no_grad():
        dog_reps = model_rep(dog_batch.to(device)).detach()
        cat_reps = model_rep(cat_batch.to(device)).detach()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Attack in steps in order to measure the effect of the step size
    cats_impersonating_dogs = cat_batch
    dogs_impersonating_cats = dog_batch

    previous_super_step = 0
    for attack_super_step in attack_super_steps:

        cats_impersonating_dogs = pgd_attack(
            model_rep,
            target_representation=dog_reps,
            origin_img=cats_impersonating_dogs,
            device=device,
            verbose=False,
            iterations=attack_super_step - previous_super_step,
            alpha=args.alpha,
            ball_size=args.ball_size,
            record_history=False,
            random_init=(
                previous_super_step == 0
            ),  # random init only for the first super step, the rest are continuation of the same attack
        )

        if args.model_name.startswith("simclr2_"):
            predictions = torch.argmax(model_linprobe(cats_impersonating_dogs, apply_fc=True), axis=1)
        else:
            predictions = torch.argmax(model_linprobe(cats_impersonating_dogs), axis=1)

        successful_dog_impersonators[attack_super_step] += [(pred in IMAGENET_DOGS_INDICES) for pred in predictions]

        dogs_impersonating_cats = pgd_attack(
            model_rep,
            target_representation=cat_reps,
            origin_img=dogs_impersonating_cats,
            device=device,
            verbose=False,
            iterations=attack_super_step - previous_super_step,
            alpha=args.alpha,
            ball_size=args.ball_size,
            record_history=False,
            random_init=(
                previous_super_step == 0
            ),  # random init only for the first super step, the rest are continuation of the same attack
        )

        if args.model_name.startswith("simclr2_"):
            predictions = torch.argmax(model_linprobe(dogs_impersonating_cats, apply_fc=True), axis=1)
        else:
            predictions = torch.argmax(model_linprobe(dogs_impersonating_cats), axis=1)

        successful_cat_impersonators[attack_super_step] += [(pred in IMAGENET_CATS_INDICES) for pred in predictions]

        previous_super_step = attack_super_step

    if idx == 0:
        impersonated_cats_sample = inverse_transform(
            torchvision.utils.make_grid(dogs_impersonating_cats[:32].cpu())
        ).resize((800, 400))
        impersonated_dogs_sample = inverse_transform(
            torchvision.utils.make_grid(cats_impersonating_dogs[:32].cpu())
        ).resize((800, 400))

        small_sample = inverse_transform(
            torchvision.utils.make_grid(
                [correct_cats_dataset.__getitem__(i)[0] for i in range(8)]
                + [cats_impersonating_dogs[i].cpu() for i in range(8)]
                + [correct_dogs_dataset.__getitem__(i)[0] for i in range(8)]
                + [dogs_impersonating_cats[i].cpu() for i in range(8)]
            )
        )

        correct_cats_sample.save(f"{args.output}_correct_cats.png", "png")
        correct_dogs_sample.save(f"{args.output}_correct_dogs.png", "png")
        impersonated_cats_sample.save(f"{args.output}_impersonated_cats.png", "png")
        impersonated_dogs_sample.save(f"{args.output}_impersonated_dogs.png", "png")
        small_sample.save(f"{args.output}_small_sample.png", "png")


for super_step in attack_super_steps:
    results[f"Cats impersonating dogs SR ({(super_step)}it)"] = float(np.mean(successful_dog_impersonators[super_step]))
    results[f"Dogs impersonating cats SR ({(super_step)}it)"] = float(np.mean(successful_cat_impersonators[super_step]))

for k, v in results.items():
    print(f"{k}: \t {100*v:.2f}%")

# save results
results.update(vars(args))
yaml_filename = args.output + ".yaml"

os.makedirs(os.path.dirname(yaml_filename), exist_ok=True)

with open(yaml_filename, "w") as f:
    yaml.dump(results, f)

print(f"Results saved to {yaml_filename}")

# save debug images
font = ImageFont.load_default()
text_img_1 = Image.new("RGB", (800, 20), color="black")
draw_1 = ImageDraw.Draw(text_img_1)
draw_1.text(
    (5, 5),
    fill="white",
    font=font,
    text=f"Correct cats:",
)

text_img_2 = Image.new("RGB", (800, 20), color="black")
draw_2 = ImageDraw.Draw(text_img_2)
draw_2.text(
    (5, 5),
    fill="white",
    font=font,
    text=f"Correct dogs:",
)

text_img_3 = Image.new("RGB", (800, 20), color="black")
draw_3 = ImageDraw.Draw(text_img_3)
draw_3.text(
    (5, 5),
    fill="white",
    font=font,
    text=f"Dogs impersonating cats:",
)

text_img_4 = Image.new("RGB", (800, 20), color="black")
draw_4 = ImageDraw.Draw(text_img_4)
draw_4.text(
    (5, 5),
    fill="white",
    font=font,
    text=f"Cats impersonating dogs:",
)

final_img = Image.new("RGB", (800, 4 * (400 + 20)))
final_img.paste(text_img_1, (0, 0))
final_img.paste(correct_cats_sample, (0, 20))
final_img.paste(text_img_2, (0, 20 + 400))
final_img.paste(correct_dogs_sample, (0, 20 + 400 + 20))
final_img.paste(text_img_3, (0, 20 + 400 + 20 + 400))
final_img.paste(impersonated_cats_sample, (0, 20 + 400 + 20 + 400 + 20))
final_img.paste(text_img_4, (0, 20 + 400 + 20 + 400 + 20 + 400))
final_img.paste(impersonated_dogs_sample, (0, 20 + 400 + 20 + 400 + 20 + 400 + 20))

final_img.save(f"{args.output}.png", "png")

print(f"Image saved to {args.output+'.png'}")
