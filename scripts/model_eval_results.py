#!/usr/bin/env python3

import os
import pandas as pd
import yaml
import tqdm
from multiprocessing import Pool
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
import seaborn as sns

DIR = "../data/model_evaluation"
OUTPUT_DIR = "evaluation_results"
ATTACK_ITERATION_MARKS = [5, 10, 30, 50]
CERT_ACC_RADII = [0.0, 0.1, 0.25, 0.75, 0.90]
ATTACK_SUBDIRS = [
    "eval_targeted_pgd_0.05",
    "eval_targeted_pgd_0.10",
    "eval_untargeted_pgd_0.05",
    "eval_untargeted_pgd_0.10",
]
NN_SUBDIRS = [
    "eval_selfsim_1.4m",
]

MODELS = {
    "ResNet50": "../data/model_evaluation/resnet",
    "PixPro": "../data/model_evaluation/pixpro",
    "AMDIM": "../data/model_evaluation/amdim",
    "MAE": "../data/model_evaluation/mae",
    "SimCLR": "../data/model_evaluation/simclr2_r50_1x_sk0",
    "SimSiam": "../data/model_evaluation/simsiam",
    "MOCO-Nonsem": "../data/model_evaluation/mocok16384_bs128_lr0.03_nonsem_noaug_16_72_nn1_alpha3",
    "MOCOv2": "../data/model_evaluation/moco",
    "MOCOv2 TAR": "../data/model_evaluation/moco_finetune_10iter_combined_pgd-targeted",
    "MOCOv2 UNTAR": "../data/model_evaluation/moco_finetune_10iter_combined_pgd-untargeted",
    "MOCOv2 L-PGD": "../data/model_evaluation/moco_finetune_10iter_combined_pgd-loss",
    "MOCOv3": "../data/model_evaluation/moco3",
    "MOCOv3 TAR": "../data/model_evaluation/moco3_finetune_10iter_combined_pgd-targeted",
    "MOCOv3 UNTAR": "../data/model_evaluation/moco3_finetune_10iter_combined_pgd-untargeted",
    "MOCOv3 L-PGD": "../data/model_evaluation/moco3_finetune_10iter_combined_pgd-loss",
}

MODEL_NAMES, MODEL_DIRS = zip(*MODELS.items())

print(f"Processing {len(MODEL_DIRS)} directories:")
for d in MODEL_DIRS:
    print(f"- {d}")
print()


def distances_to_universal_quantiles(quantiles_dict, norm, distance):
    """
    Convert distances to universal quantiles provided a given distribution via quantiles_dict.
    """
    if distance is not None:
        sign = np.sign(distance)
        res_key, _ = min(quantiles_dict[norm].items(), key=lambda x: abs(abs(distance) - x[1]))
        return float(sign * res_key)
    else:
        return None


def disance_norms_at_iter(quantiles_dict, info_dict, iteration, prefix):
    """
    Compute universal quantiles at a given attack iteration.
    """
    return_dict = {}
    for dist_size_type in ["l1", "l2", "linf"]:
        return_dict[f"{prefix} {dist_size_type} U"] = distances_to_universal_quantiles(
            quantiles_dict, dist_size_type, info_dict[f"dist_{dist_size_type}"][iteration]
        )
    return return_dict


def relative_at_iter(info_dict, iteration, prefix):
    """
    Compute relative quantiles at a given attack iteration.
    """
    return_dict = {}
    for dist_size_type in ["l1", "l2", "linf"]:
        return_dict[f"{prefix} {dist_size_type} R"] = (
            info_dict[f"dist_{dist_size_type}"][iteration] / info_dict[f"dist_{dist_size_type}"][0]
        )
    return return_dict


def normalized_at_iter(info_dict, iteration, prefix, medians):
    """
    Compute normalized (by the median) distances at a given attack iteration.
    """
    return_dict = {}
    for dist_size_type in ["l1", "l2", "linf"]:
        return_dict[f"{prefix} {dist_size_type} N"] = (
            info_dict[f"dist_{dist_size_type}"][iteration] / medians[dist_size_type]
        )
    return return_dict


def make_a_row_attacks(args):
    """
    Processes a single adversarial attack.
    """

    info_file, quantiles_dict = args
    with open(info_file, "r") as f:
        info_dict = yaml.safe_load(f)

    return_dict = {
        "alpha": info_dict.get("alpha", None),
        "ball_size": info_dict.get("ball_size", None),
        "attack_": info_dict.get("attack", None),
        "model": info_dict.get("model", None),
        "origin_name": info_dict.get("origin_name", None),
        "target_name": info_dict.get("target_name", None),
        "batch_size": info_dict.get("batch_size", None),
        "model_weights_path": info_dict.get("model_weights_path", None),
        "targeted": info_dict.get("targeted", None),
    }

    if info_dict["attack"] == "pgd" and info_dict["targeted"]:
        return_dict["attack"] = f"Targeted PGD ({info_dict['ball_size']:.2f})"
    elif info_dict["attack"] == "pgd" and not info_dict["targeted"]:
        return_dict["attack"] = f"Untargeted PGD ({info_dict['ball_size']:.2f})"
    elif info_dict["attack"] == "fgsm" and info_dict["targeted"]:
        return_dict["attack"] = "Targeted FGSM"
    elif info_dict["attack"] == "fgsm" and not info_dict["targeted"]:
        return_dict["attack"] = "Untargeted FGSM"

    medians = {
        "l1": min(quantiles_dict["l1"].items(), key=lambda x: abs(0.5 - x[0]))[1],
        "l2": min(quantiles_dict["l2"].items(), key=lambda x: abs(0.5 - x[0]))[1],
        "linf": min(quantiles_dict["linf"].items(), key=lambda x: abs(0.5 - x[0]))[1],
    }

    for i in ATTACK_ITERATION_MARKS:
        prefix = f"{i}it"

        # not i-1 bellow as the history records the attack state BEFORE the optimizer takes a step
        return_dict.update(disance_norms_at_iter(quantiles_dict, info_dict, i, prefix))
        return_dict.update(normalized_at_iter(info_dict, i, prefix, medians))

        if return_dict["attack"].startswith("Targeted"):
            return_dict.update(relative_at_iter(info_dict, i, prefix))

        if "cosine_similarity" in info_dict:
            return_dict[f"{prefix} cosine_similarity"] = info_dict["cosine_similarity"][i]

    return return_dict


certified_accuracy_plots = []
certified_robustness_quantiles_plots = []
certified_robustness_normalized_plots = []
all_margin_ratios = []

def process_directory(model_name, d):
    """
    Process the results of a single model.
    """
    results = dict()

    ## LOAD THE QUANTILES
    quantiles_path = os.path.join(d, "quantiles.yaml")
    if not os.path.exists(quantiles_path):
        print(f"QUANTILES FOR {d} DO NOT EXIST!")
        return None

    with open(quantiles_path, "r") as f:
        quantiles_dict = yaml.safe_load(f)

    l1_qs, l1_vs = zip(*quantiles_dict["l1"].items())
    l2_qs, l2_vs = zip(*quantiles_dict["l2"].items())
    linf_qs, linf_vs = zip(*quantiles_dict["linf"].items())
    print("Quantiles loaded")

    # ## MAKE PLOT WITH THE DISTANCE DISTRIBUTIONS
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    # fig.suptitle(f"Distribution of distances between representations")
    # ax1.plot(l1_qs, l1_vs)
    # ax1.set_title("$\ell_1$ distance distribution")
    # ax2.plot(l2_qs, l2_vs)
    # ax2.set_title("$\ell_2$ distance distribution")
    # ax3.plot(linf_qs, linf_vs)
    # ax3.set_title("$\ell_\inf$ distance distribution")
    # ax1.grid()
    # ax2.grid()
    # ax3.grid()
    # plt.savefig(os.path.join(d, "quantiles.png"))
    # print("Distribution of distances between representations plot done")

    # OBTAIN MEDIAN PERCENTILES OF ATTACKS
    filenames_to_process = []
    for attacks_dir in ATTACK_SUBDIRS:
        for path, subdirs, files in os.walk(os.path.join(d, attacks_dir)):
            for file in files:
                if file.endswith(".info"):
                    filenames_to_process.append(os.path.join(path, file))
    if len(filenames_to_process) > 0:
        with Pool(32) as p:
            row_list = list(
                tqdm.tqdm(
                    p.imap(make_a_row_attacks, zip(filenames_to_process, [quantiles_dict] * len(filenames_to_process))),
                    total=len(filenames_to_process),
                    desc="Attacks",
                )
            )
        df = pd.DataFrame(row_list)

        # assert that only comparable entries are considered
        assert len(np.unique(df["model"].value_counts)) == 1
        assert len(np.unique(df["batch_size"].value_counts)) == 1
        assert len(np.unique(df["model_weights_path"].value_counts)) == 1
        assert len(np.unique(df["alpha"].value_counts)) == 1

        data = df.groupby(["attack"]).median()
        data["N"] = df.groupby(["attack"]).size()

        df_out = data.stack()
        df_out.index = df_out.index.map("{0[0]} {0[1]}".format)
        df_out.to_frame().T

        results.update(df_out.to_dict())

    ## OBTAIN ACCURACY INFORMATION FROM THE LINEAR PROBE RESULTS
    lin_probe_path = os.path.join(d, "linear_probe_results.yaml")
    if os.path.exists(lin_probe_path):
        with open(lin_probe_path, "r") as f:
            t = yaml.safe_load(f)
            results["Top-1 accuracy"] = t.get("Top-1 accuracy", None)
            results["Top-5 accuracy"] = t.get("Top-5 accuracy", None)

            # we will convert to percentages later
            if "Top-1 accuracy" in results:
                results["Top-1 accuracy"] /= 100
            if "Top-5 accuracy" in results:
                results["Top-5 accuracy"] /= 100
        print("Linear probe results processed")

    ## OBTAIN ACCURACY INFORMATION FROM THE LOWPASS LINEAR PROBE RESULTS
    lin_probe_lowpass_path = os.path.join(d, "linear_probe_lowpass_results.yaml")
    if os.path.exists(lin_probe_lowpass_path):
        with open(lin_probe_lowpass_path, "r") as f:
            t = yaml.safe_load(f)
            results["Lowpass Top-1 accuracy"] = t.get("Top-1 accuracy", None)
            results["Lowpass Top-5 accuracy"] = t.get("Top-5 accuracy", None)

            # we will convert to percentages later
            if "Lowpass Top-1 accuracy" in results:
                results["Lowpass Top-1 accuracy"] /= 100
            if "Lowpass Top-5 accuracy" in results:
                results["Lowpass Top-5 accuracy"] /= 100
        print("Low pass linear probe results processed")


    ## OBTAIN MARGIN METRICS
    margin_path = os.path.join(d, "margin.csv")
    if os.path.exists(margin_path):
        margins_df = pd.read_csv(margin_path)
        margins_df["Margin ratio"] = (margins_df["d_x_xptox"] - margins_df["d_x_xtoxp"]) / margins_df["d_x_xp"]
        margins_df["Model"] = model_name
        margins_df["Clean divergence"] = [
            distances_to_universal_quantiles(quantiles_dict, "l2", d) 
            for d in margins_df["d_x_xp"]
        ]
        
        all_margin_ratios.append(margins_df)
        
        results["Mean margin ratio"] = margins_df["Margin ratio"].mean()
        results["Median margin ratio"] = margins_df["Margin ratio"].median()
        results["Overlap risk"] = (margins_df["Margin ratio"]<0).mean()
        
        print("Margin results processed")

    ## OBTAIN NEAREST NEIGHBOURS METRICS
    for dir in NN_SUBDIRS:
        for path, subdirs, files in os.walk(os.path.join(d, dir)):
            for file in files:
                if file.endswith(".info"):
                    with open(os.path.join(path, file), "r") as f:
                        t = yaml.safe_load(f)
                        n_candidates_str = {
                            1000: "1k",
                            10000: "10k",
                            100000: "100k",
                            1000000: "1m",
                            1440191: "1.4m",
                        }.get(t.get("n_candidates"), "?")
                        results[f"NN Top-1 accuracy ({n_candidates_str})"] = t.get("acc_top1", None)
                        results[f"NN Top-5 accuracy ({n_candidates_str})"] = t.get("acc_top5", None)
                        results[f"Break away risk ({n_candidates_str})"] = t.get("risk", None)
    print("Nearest neighbours results processed")

    ## OBTAIN CERTIFIED ACCURACY
    cert_acc_path = os.path.join(d, "noisy_certified_accuracy.csv")
    if os.path.exists(cert_acc_path):
        acc_path = pd.read_csv(cert_acc_path)
        acc_path.columns = acc_path.columns.str.strip()
        for radius in CERT_ACC_RADII:
            results[f"CertAcc @ {radius:.2f}"] = len(
                acc_path[(acc_path["correct"] == 1) & (acc_path["radius"] > radius)]
            ) / len(acc_path)

        # Average Certified Radius as per MACER by Zhai et al (ICLR 2020)
        # https://github.com/RuntianZ/macer/blob/master/rs/certify.py
        results[f"Average Certified Radius"] = acc_path[acc_path["correct"] == 1]["radius"].mean()

        radii = acc_path["radius"].unique()
        radii.sort()
        for radius in radii:
            certified_accuracy_plots.append(
                {
                    "Model": model_name,
                    "Radius": radius,
                    "Accuracy": len(acc_path[(acc_path["correct"] == 1) & (acc_path["radius"] > radius)])
                    / len(acc_path),
                }
            )
        print("Certified accuracy results processed")

    ## OBTAIN CERTIFIED ROBUSTNESS
    cert_rob_path = os.path.join(d, "certified_robustness.csv")
    if os.path.exists(cert_rob_path):
        cert_rob = pd.read_csv(cert_rob_path)
        cert_rob.columns = cert_rob.columns.str.strip()

        normalize = lambda v: v / min(quantiles_dict["l2"].items(), key=lambda x: abs(0.5 - x[0]))[1]

        cert_rob["eps_out_quantiles"] = [
            distances_to_universal_quantiles(quantiles_dict, "l2", d) for d in cert_rob["eps_out"]
        ]
        cert_rob["eps_out_normalized"] = [normalize(d) for d in cert_rob["eps_out"]]
        if len(cert_rob) > 0:
            qs = cert_rob["eps_out_quantiles"].unique()
            qs.sort()
            for q in qs:
                certified_robustness_quantiles_plots.append(
                    {
                        "Model": model_name,
                        "Quantile": q,
                        "Fraction": float(len(cert_rob[cert_rob["eps_out_quantiles"] > q])) / len(cert_rob),
                    }
                )

            nvs = cert_rob["eps_out_normalized"].unique()
            nvs.sort()
            for v in nvs:
                certified_robustness_normalized_plots.append(
                    {
                        "Model": model_name,
                        "Normalized radius": v,
                        "Fraction": float(len(cert_rob[cert_rob["eps_out_normalized"] > v])) / len(cert_rob),
                    }
                )
        print("Certified robustness results processed")

    ## OBTAIN IMPERSONATION RESULTS
    impersonation_path = os.path.join(d, "impersonation.yaml")
    if os.path.exists(impersonation_path):
        with open(impersonation_path, "r") as f:
            t = yaml.safe_load(f)
            results["Cat detection accuracy"] = t.get("Cat detection accuracy", None)
            results["Dog detection accuracy"] = t.get("Dog detection accuracy", None)

            results.update({k: v for k, v in t.items() if "impersonating" in k})

            # Add the balanced impersonation success
            for k, v in t.items():
                if "Dogs impersonating cats SR" in k:
                    iterations = k[len("Dogs impersonating cats SR ") :]
                    if f"Dogs impersonating cats SR {iterations}" in t:
                        results[f"Impersonation success rate {iterations}"] = 0.5 * (
                            results[f"Dogs impersonating cats SR {iterations}"]
                            + results[f"Cats impersonating dogs SR {iterations}"]
                        )

        print("Impersonation results processed")

    return results


results = list()
names = list()
for d, n in zip(MODEL_DIRS, MODEL_NAMES):
    print(f"STARTING PROCESSING {d}")
    r = process_directory(n, d)
    if r is not None:
        print(f"Processing {d} finished!")
        results.append(r)
        names.append(n)
    else:
        print(f"FAILURE in processing of {d}!!!")

    print("")

os.makedirs(OUTPUT_DIR, exist_ok=True)
df_total = pd.DataFrame(results, index=names)
df_total.to_csv(os.path.join(OUTPUT_DIR, "RESULTS.csv"))

def rename_models(df):
    """Splits the model name in two parts for cleaner plotting."""
    df["Training"] = df["Model"].str.split(expand=True)[1].fillna('Original')
    df["Model"] = df["Model"].str.split(expand=True)[0]
    return df

# make the certified accuracy plot
if len(certified_accuracy_plots) > 0:
    plot_data = rename_models(pd.DataFrame(certified_accuracy_plots))
    plt.clf()
    plot = sns.lineplot(
        x="Radius",
        y="Accuracy",
        hue="Model",
        style="Training",
        data=plot_data,
        markers=False,
    )
    plot.legend().get_frame().set_linewidth(0.0)
    plot.set_title("Certified accuracy")
    sns.move_legend(plot, "upper right", ncol=2, frameon=False)  
    sns.despine()
    plot.figure.savefig(os.path.join(OUTPUT_DIR, "certified_accuracy.png"))

# make the certified robustness plot (quantiles)
if len(certified_robustness_quantiles_plots)>0:
    plot_data = rename_models(pd.DataFrame(certified_robustness_quantiles_plots))
    plt.clf()
    plot = sns.lineplot(
        x="Quantile",
        y="Fraction",
        hue="Model",
        style="Training",
        data=plot_data,
        markers=False,
        palette="colorblind",
    )
    plot.legend().get_frame().set_linewidth(0.0)
    plot.set_title("Certified robustness")
    plot.set(xscale="log")
    sns.move_legend(plot, "upper right", ncol=2, frameon=False)  
    sns.despine()
    plot.figure.savefig(os.path.join(OUTPUT_DIR, "certified_robustness_quantiles.png"))

# make the certified robustness plot (normalized)
if len(certified_robustness_normalized_plots) > 0:
    plot_data = rename_models(pd.DataFrame(certified_robustness_normalized_plots))
    plt.clf()
    plot = sns.lineplot(
        x="Normalized radius",
        y="Fraction",
        hue="Model",
        style="Training",
        data=plot_data,
        markers=False,
        palette="colorblind",
    )
    plot.legend().get_frame().set_linewidth(0.0)
    plot.set_title("Certified robustness")
    plot.set(xscale="log")
    sns.move_legend(plot, "upper right", ncol=2, frameon=False)  # bbox_to_anchor=(0.5, 1.15),
    sns.despine()
    plot.figure.savefig(os.path.join(OUTPUT_DIR, "certified_robustness_normalized.png"))

#　make the margin ratio plot
if len(all_margin_ratios) > 0:
    plot_data = rename_models(pd.concat(all_margin_ratios, ignore_index=True))
    plot_data["Clean divergence"] = plot_data["Clean divergence"].round(2)
    plt.clf()
    plot = sns.lineplot(
        x="Clean divergence",
        y="Margin ratio",
        hue="Model",
        style="Training",
        data=plot_data,
        markers=False,
        palette="colorblind",
    )
    plot.legend().get_frame().set_linewidth(0.0)
    sns.despine()
    plot.figure.savefig(os.path.join(OUTPUT_DIR, "margin_ratio.png"))
    