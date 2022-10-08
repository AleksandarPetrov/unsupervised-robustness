# import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm

from .tools import AttackHistory, unsqueeze_if_necessary


def pgd_attack(
    model,
    target_representation,
    origin_img,
    device,
    verbose=True,
    iterations=5,
    ball_size=0.1,
    loss_fn=nn.MSELoss(reduction="none"),
    alpha=0.005,
    record_history=False,
    random_init=True,
    normalization_fn=None,
    amp=False,
):

    if normalization_fn is None:
        normalization_fn = lambda x: x

    assert loss_fn.reduction == "none", "Reduction must be none!"

    if record_history:
        attack_histories = [AttackHistory() for _ in range(origin_img.size(dim=0))]

    inversion_img = unsqueeze_if_necessary(origin_img).clone().detach().to(device, non_blocking=True)

    # if target_representation is None, then we perform an untargeted attack
    if target_representation is not None:
        target_representation = target_representation.clone().detach().to(device, non_blocking=True)
        targeted = True
    else:
        with torch.no_grad():
            if amp:
                with torch.cuda.amp.autocast(True):
                    target_representation = model(normalization_fn(inversion_img))
            else:
                target_representation = model(normalization_fn(inversion_img))
        targeted = False

    if random_init:
        perturbation = torch.empty_like(inversion_img).uniform_(-ball_size, ball_size)
    else:
        perturbation = torch.zeros_like(inversion_img)

    perturbation.requires_grad = True

    if verbose:
        pbar = tqdm.tqdm(range(iterations))
    else:
        pbar = range(iterations)
    for i in pbar:
        # Compute prediction and loss
        if amp:
            with torch.cuda.amp.autocast(True):
                pred = model(normalization_fn(inversion_img + perturbation))
        else:
            pred = model(normalization_fn(inversion_img + perturbation))

        individual_losses = loss_fn(pred, target_representation).sum(dim=1)
        loss = individual_losses.sum()
        l2_norms = torch.sqrt(individual_losses)

        # Backpropagation
        loss.backward()
        model.zero_grad()

        with torch.no_grad():
            sign_data_grad = perturbation.grad.sign()

            # minimize loss (distance to target representation) if targeted and
            # maximize (distance from initial representation) if untargeted
            perturbation.add_((-1 if targeted else 1) * alpha * sign_data_grad)
            perturbation.clamp_(min=-ball_size, max=ball_size)
            perturbation.grad.data.zero_()
            torch.maximum(perturbation, 0 - inversion_img, out=perturbation)
            torch.minimum(perturbation, 1 - inversion_img, out=perturbation)

        if verbose:
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "worst l2": l2_norms.max().item() if targeted else l2_norms.min().item(),
                }
            )

        if record_history:
            with torch.no_grad():
                diff_vector = torch.abs(pred - target_representation)
                flat_perturbation = torch.abs(perturbation).flatten(start_dim=1)

                l1_norms = torch.sum(diff_vector, dim=1)
                l2_norms = torch.sqrt(torch.sum(diff_vector**2, dim=1))
                linf_norms, _ = torch.max(diff_vector, dim=1)

                perturb_l1 = torch.sum(flat_perturbation, dim=1)
                perturb_l2 = torch.sqrt(torch.sum(flat_perturbation**2, dim=1))
                perturb_linf, _ = torch.max(flat_perturbation, dim=1)

                cosine_similarity = torch.nn.CosineSimilarity(dim=1)(pred, target_representation)

                for idx in range(origin_img.size(dim=0)):
                    attack_histories[idx].dist_l1.append(l1_norms[idx].item())
                    attack_histories[idx].dist_l2.append(l2_norms[idx].item())
                    attack_histories[idx].dist_linf.append(linf_norms[idx].item())

                    attack_histories[idx].perturb_l1.append(perturb_l1[idx].item())
                    attack_histories[idx].perturb_l2.append(perturb_l2[idx].item())
                    attack_histories[idx].perturb_linf.append(perturb_linf[idx].item())

                    attack_histories[idx].cosine_similarity.append(cosine_similarity[idx].item())

    if record_history:
        return inversion_img + perturbation, attack_histories
    else:
        return inversion_img + perturbation
