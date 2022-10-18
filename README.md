# Robustness of Unsupervised Representation Learning without Labels

This is the official implementation of the experiments in our [Robustness of Unsupervised Representation Learning without Labels](https://arxiv.org/abs/2210.04076) paper.
## Installation

This section outlines the models and datasets that need to be downloaded as well as the Python packages needed to reproduce the results.

### Downloading ImageNet

ImageNet must be downloaded from [the official website](https://image-net.org/download-images.php). 
We use the ImageNet Large Scale Visual Recognition Challenge 2012.
You need `Training images (Task 1 & 2).`(`ILSVRC2012_img_train.tar`) and `Validation images (all tasks)`(`ILSVRC2012_img_val.tar`).
Download the two archives to `data/imagenet`.

Then unpack the training data with:

```
cd data/imagenet
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
```

And the validation data with:

```
cd data/imagenet
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

### Downloading PASS

To download [PASS](https://www.robots.ox.ac.uk/~vgg/data/pass/), just run the `PASS-download.sh` utility script in `data`:

```
cd data && ./PASS-download.sh
```

### Downloading the Assira Pets images dataset

You can download the Assira dataset from [here](https://www.microsoft.com/en-us/download/details.aspx?id=54765) .
Some of the files are corrupt and cause the evaluation to fail. Delete them with:

```
rm -f PetImages/Cat/666.jpg PetImages/Dog/11702.jpg
```

### Downloading the  baseline models

 You can download the baseline models that we evaluate from their official sources.
 For convenience, we provide links below. 
 Place the models in `/data/models/`:

- [PixPro 400 epochs](https://drive.google.com/file/d/1Ox2RoFbTrrllbwvITdZvwkNnKUQSUPmV/view)
- [MAE  ViT-Large](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth)
- [MOCOv2 200 epochs](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar)
- [MOCOv3 ViT-Small](https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar)
- [Contrastive Learning with Non-Semantic Negatives +Patch, k=16384, α=3](https://drive.google.com/file/d/1w_FgptIAfFHjGQCxTAATkHKw-9_CDZUu/view)
- [AMDIM (Medium)](https://amdimmodels.blob.core.windows.net/amdim/amdim_ndf256_rkhs2048_rd10.pth)
- [SimSiam (100 epochs, batch size 256](https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar), then renamed to `simsiam_100ep_256bs.pth.tar`
- [SimCLRv2 (depth 50, width 1x, without selective kernels)](https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_1x.zip) needs to be [converted to PyTorch](https://github.com/Separius/SimCLRv2-Pytorch). We provide the converted model [here](https://drive.google.com/file/d/1usdcrdGZBoAGxGO9itbS2LuoAy5pD2C0/view?usp=sharing).

### Downloading the adversarially fine-tuned models

We are also sharing the adversarially fine-tuned [MOCOv2](https://github.com/facebookresearch/moco) and [MOCOv3](https://github.com/facebookresearch/moco-v3) models.
Download the archives to `data` and unzip them.
Each of the archives contains the three adversarially fine-tuned versions: with targeted attacks, untargeted attacks, and Ĺ-PGD attacks.

- [Adversarially fine-tuned MOCOv2](https://drive.google.com/file/d/1qP4bQSOJlja8jqGqj1SLxN7CUrwhYwlE/view?usp=sharing)
- [Adversarially fine-tuned MOCOv3](https://drive.google.com/file/d/1xxZWBD_Kh1tcKiPekipqfmN5aTdXLqCB/view?usp=sharing)

### Linear probes and intermediate evaluation files

We provide three linear probes for each of the baseline and fine-tuned models: standard, with Gaussian noise, and low-pass.
Furthermore, we offer the intermediate states for the model evaluation, allowing for a quick verification of our results.
To obtain the linear probes and the intermediate evaluation files, download the following archives to `data` and unzip them.

- [AMDIM](https://drive.google.com/file/d/1iRdhkEMctdMV7FWiKXIEgMU_5G1qRXZV/view?usp=sharing)
- [MAE](https://drive.google.com/file/d/1TA371lwo6ZjN2gwPweRBSd7IGCicLI14/view?usp=sharing)
- [MOCOv2](https://drive.google.com/file/d/1jAqRajnmcyHOF-YpVtW6Vx0PVRFsvikq/view?usp=sharing)
- [MOCOv2 Untargeted](https://drive.google.com/file/d/1MXPVep_ZA6uDTZr82458ENTsGhXjALDT/view?usp=sharing)
- [MOCOv2 Targeted](https://drive.google.com/file/d/1qN5tbokJ4L91D6bT4Wgu-aIH68TksEg6/view?usp=sharing)
- [MOCOv2 Ĺ-PGD](https://drive.google.com/file/d/1fCEan_kwQ3WQzRsrDE4Y6u33JIPGSk9U/view?usp=sharing)
- [MOCOv3](https://drive.google.com/file/d/1OV4-9O7DE-WVUct4dbdcs1AFDCJzFmTx/view?usp=sharing)
- [MOCOv3 Untargeted](https://drive.google.com/file/d/1EA697w-W6LsnprlSJnvIYXAQAzxwqggm/view?usp=sharing)
- [MOCOv3 Targeted](https://drive.google.com/file/d/1V50T6_LXz8xgW2nOvWmhOot6yVxRbdX8/view?usp=sharing)
- [MOCOv3 Ĺ-PGD](https://drive.google.com/file/d/1J7EuKhp7brw0gvSffCrQTAroDeKUVu14/view?usp=sharing)
- [Contrastive Learning with Non-Semantic Negatives +Patch, k=16384, α=3](https://drive.google.com/file/d/1n-KGSBrRP8XZ9cmkqBAPuTbh3ilSJjA0/view?usp=sharing)
- [PixPro](https://drive.google.com/file/d/15QH5ZN-Dw3QxtV4GLnmzzBtz1mnI1Iqf/view?usp=sharing)
- [ResNet](https://drive.google.com/file/d/10A5eiYXEAMBs5FRq0tzpsQttRNftuUWy/view?usp=sharing)
- [SimCLRv2](https://drive.google.com/file/d/1WFdZgiaQW8dwHBfl0VBzfMLG5Aop62Ua/view?usp=sharing)
- [SimSiam](https://drive.google.com/file/d/13p53IS8hpNjIu331uHstZw9juMWfbbIZ/view?usp=sharing)

### Resulting folder structure

If you have chosen to download all of the above, the resulting structure of the `data` directory should look like:

```
data
├ imagenet
  ├ train
    ├ n01440764
      ...
    └ n15075141
  └ val
    ├ n01440764
      ...
    └ n15075141  

├ model_evaluation  
  ├ amdim
    ├ certified_robustness.csv
    ├ eval_selfsim_1.4m
      └ ... 
    ├ eval_targeted_pgd_0.05
      └ ... 
    ├ eval_targeted_pgd_0.10
      └ ... 
    ├ eval_untargeted_pgd_0.05
      └ ... 
    ├ eval_untargeted_pgd_0.10
      └ ... 
    ├ impersonation.png
    ├ impersonation.yaml
    ├ impersonation_correct_cats.png
    ├ impersonation_correct_dogs.png
    ├ impersonation_impersonated_cats.png
    ├ impersonation_impersonated_dogs.png
    ├ impersonation_small_sample.png
    ├ lin_probe_lowpass_weights
      └ ... 
    ├ lin_probe_noisy_weights
      └ ... 
    ├ lin_probe_weights
      └ ... 
    ├ linear_probe_lowpass_results.yaml
    ├ linear_probe_results.yaml
    ├ margin.csv
    ├ noisy_certified_accuracy.csv
    ├ quantiles.yaml
    ├ training_set_sample.png
    └ validation_set_sample.png

  ├ mae 
    └ ... 
  ├ moco
    └ ... 
  ├ moco_finetune_10iter_combined_pgd-loss
    └ ... 
  ├ moco_finetune_10iter_combined_pgd-targeted
    └ ... 
  ├ moco_finetune_10iter_combined_pgd-untargeted
    └ ... 
  ├ moco3
    └ ... 
  ├ moco3_finetune_10iter_combined_pgd-loss
    └ ... 
  ├ moco3_finetune_10iter_combined_pgd-targeted
    └ ... 
  ├ moco3_finetune_10iter_combined_pgd-untargeted
    └ ... 
  ├ mocok16384_bs128_lr0.03_nonsem_noaug_16_72_nn1_alpha3
    └ ... 
  ├ pixpro
    └ ... 
  ├ resnet
    └ ... 
  ├ simclr2_r50_1x_sk0
    └ ... 
  └ simsiam 
    └ ... 

├ models
  ├ finetune_loss_upgd
    ├ checkpoint_0209.pth.tar
    ├ checkpoint_0209.pth.tar.args
    ├ training_set_sample_epoch209_0.png
    ├ training_set_sample_epoch209_1.png
    └ training_set_sample_epoch209_attacked.png
 
  ├ finetune_targeted_upgd
    └ ... 
  ├ finetune_untargeted_upgd
    └ ... 
  ├ moco3_finetune_loss_upgd
    └ ... 
  ├ moco3_finetune_targeted_upgd
    └ ... 
  ├ moco3_finetune_untargeted_upgd
    └ ... 
  ├ amdim_ndf256_rkhs2048_rd10.pth
  ├ mae_pretrain_vit_large.pth
  ├ moco_v2_200ep_pretrain.pth.tar
  ├ mocov2_mocok16384_bs128_lr0.03_nonsem_noaug_16_72_nn1_alpha3_epoch200.pth.tar
  ├ pixpro_base_r50_400ep_md5_919c6612.pth
  ├ simclr2_r50_1x_sk0.pth
  ├ simsiam_100ep_256bs.pth.tar
  └ vit-s-300ep.pth.tar
   
├ PASS dataset
  ├ 0
  ├ 1
    ...
  └ 19
└ PetImages
  ├ Cat
  └ Dog

```

<!-- You can verify that you have all the data with the following command:

```
find . -type f | grep -o ".[^.]\+$" | sort | uniq -c
```

You should have the following number of images in the various folders:

| **Directory**  | **Number of images** |
|----------------|---------------------:|
| imagenet/train |            1,281,167 |
| imagenet/val   |               50,000 |
| PASS_dataset   |            1,440,191 |
| PetImages/Cat  |               12,499 |
| PetImages/Dog  |               12,499 | -->

### Installing the necessary Python pacakges

You can use the provided `requirements.txt`:

```
pip install -r requirements.txt
```

### Cloning the randomized smoothing code 

We use the randomized smoothing implementation by the [CMU Locus Lab](https://github.com/locuslab/smoothing).
It is a submodule of this repository, so run the following commands to clone it:

```
git submodule init
git submodule update
```

## Adversarial fine-tuning of MOCOv2 and MOCOv3

We provide a modification of the [MOCOv2 training code](https://github.com/facebookresearch/moco) with added adversarial fine-training capabilities.
You can train the models using the convenience script in the `moco` directory:

```
cd moco && ./MOCO_experiments.sh
```

Similarly, we fine-tuned MOCOv3 using a modifcation of its [original training code](https://github.com/facebookresearch/moco-v3).

```
cd moco-v3 && ./MOCO3_experiments.sh
```

The batchsizes are optimized to run on 4 GeForce RTX 2080 Ti GPUs with 11GB memory and may have to be adjusted for other devices.
Alternatively, you can download the trained models as explained above.

## Training the linear probes

We provide the best checkpoints for all linear probes above so you don't need to train them again.

Should you wish to do it though, the linear probes can be trained using the `run_pipeline` utility tool.
It manages the training of all of them, taking into account the number of GPUs you have.
This is the easiest way to do it but if you want to train individual linear probes you can use the `--dry-run` option to see their names and the `--single-tasks` option to run one or more of them (separated with commas).
Alternatively, the `--just-show` will show the actual command used to train each model.


To train all linear probes (substititute with the GPU ids you want to use):

```
scripts/run_pipeline.py --gpus 0,1,2,3  scripts/linprobe_training
```

To train only some of them (amend the list provided to `--single-tasks` to fit your needs):
```
scripts/run_pipeline.py --gpus 0,1,2,3 --single-tasks moco/train_linpro,simclr-sk0/train_linpro_noisy scripts/linprobe_training
```

To see the commands used for training:
```
scripts/run_pipeline.py --gpus 0,1,2,3 --single-tasks moco/train_linpro,simclr-sk0/train_linpro_noisy --just-show scripts/linprobe_training
```

Should there be any errors, you can find the logs of the individual tasks under `scripts/linprobe_training_data_/logs`.

## Evaluating the models

We have performed a large number of evaluations for every single model.
In order to manage the volume of tasks we also used the `run_pipeline` utility.
This can take considerable time, so we recommend using the intermediate evaluation files we provide above.

Then, generating a CSV with all results is as easy as:

```
cd scripts & ./model_eval_results.py
```

The results would be in the `scripts/evaluation_results` directory.

Should you prefer to do the complete evaluation yourself or with different parameters, you can do it with:

```
scripts/run_pipeline.py --gpus 0,1,2,3  scripts/model_evaluation
```

You can use the `--single-tasks`, `--just-show` and `--dry-run` options as described before.

## Contributors

- [Aleksandar Petrov](https://p-petrov.com)
- [Marta Kwiatkowska](https://www.cs.ox.ac.uk/people/marta.kwiatkowska/)

## Citation

If you find this work useful for your research, please cite it as:

```
@article{
    petrov2022robustness,
    title={Robustness of Unsupervised Representation Learning without Labels},
    author={Aleksandar Petrov and Marta Kwiatkowska},
    journal = {arXiv preprint arXiv:2202.01181},
    year={2022},
}
```
