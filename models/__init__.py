from .simclr_v2 import (
    simCLR_get_model,
    simCLR_preprocessing_transform,
    simCLR_inverse_transform,
    simCLR_training_augmentation,
)

from .moco_v2 import (
    moco_get_model,
    moco_preprocessing_transform,
    moco_inverse_transform,
    moco_training_augmentation,
)

from .simsiam import (
    simsiam_get_model,
    simsiam_preprocessing_transform,
    simsiam_inverse_transform,
    simsiam_training_augmentation,
)

from .pixpro import (
    pixpro_get_model,
    pixpro_preprocessing_transform,
    pixpro_inverse_transform,
    pixpro_training_augmentation,
)

from .amdim import (
    amdim_get_model,
    amdim_preprocessing_transform,
    amdim_inverse_transform,
    amdim_training_augmentation,
)

from .mae import (
    mae_get_model,
    mae_preprocessing_transform,
    mae_inverse_transform,
    mae_training_augmentation,
    NativeScalerWithGradNormCount,
)

from .moco_nonsem import (
    moco_nonsem_get_model,
    moco_nonsem_preprocessing_transform,
    moco_nonsem_inverse_transform,
    moco_nonsem_training_augmentation,
)

from .resnet import (
    resnet_get_model,
    resnet_preprocessing_transform,
    resnet_inverse_transform,
    resnet_training_augmentation,
)

from .moco_v3 import (
    moco3_get_model,
    moco3_preprocessing_transform,
    moco3_inverse_transform,
    moco3_training_augmentation,
)

import torch

def get_model(model_name, model_weights_path, for_lin_probe=False, resnet_fc_init=False):

    # Load the model and its corresponding transforms
    if model_name == "moco-v2":
        if for_lin_probe:
            model = moco_get_model(model_weights_path, num_classes=1000)

            # freeze all layers but the last fc
            for name, param in model[1].named_parameters():
                if name not in ["fc.weight", "fc.bias"]:
                    param.requires_grad = False
            # init the fc layer
            model[1].fc.weight.data.normal_(mean=0.0, std=0.01)
            model[1].fc.bias.data.zero_()

        else:
            model = moco_get_model(model_weights_path, num_classes=128)

        preprocessing_transform = moco_preprocessing_transform
        inverse_transform = moco_inverse_transform
        training_augmentation = moco_training_augmentation

    elif model_name == "moco-v3":
        if for_lin_probe:
            model = moco3_get_model(model_weights_path)
            # freeze all layers but the last fc
            for name, param in model[1].named_parameters():
                if name not in ["head.weight", "head.bias"]:
                    param.requires_grad = False
            # create and init the fc layer
            final_layer_input_dim = model[1].head[6].weight.shape[1]
            del model[1].head[7], model[1].head[6]
            model[1].head.append(torch.nn.Linear(final_layer_input_dim, 1000, bias=True))
            model[1].head[6].weight.data.normal_(mean=0.0, std=0.01)
            model[1].head[6].bias.data.zero_()
 
        else:
            model = moco3_get_model(model_weights_path, num_classes=256)

        preprocessing_transform = moco3_preprocessing_transform
        inverse_transform = moco3_inverse_transform
        training_augmentation = moco3_training_augmentation

    elif model_name == "moco-nonsem":
        if for_lin_probe:
            model = moco_nonsem_get_model(model_weights_path, num_classes=1000)

            # freeze all layers but the last fc
            for name, param in model[1].named_parameters():
                if name not in ["fc.weight", "fc.bias"]:
                    param.requires_grad = False
            # init the fc layer
            model[1].fc.weight.data.normal_(mean=0.0, std=0.01)
            model[1].fc.bias.data.zero_()

        else:
            model = moco_nonsem_get_model(model_weights_path, num_classes=1000)

        preprocessing_transform = moco_nonsem_preprocessing_transform
        inverse_transform = moco_nonsem_inverse_transform
        training_augmentation = moco_nonsem_training_augmentation
    elif model_name == "simsiam":
        if for_lin_probe:
            model = simsiam_get_model(model_weights_path, num_classes=1000)

            # freeze all layers but the last fc
            for name, param in model[1].named_parameters():
                if name not in ["fc.weight", "fc.bias"]:
                    param.requires_grad = False
            # init the fc layer
            model[1].fc.weight.data.normal_(mean=0.0, std=0.01)
            model[1].fc.bias.data.zero_()

        else:
            model = simsiam_get_model(model_weights_path, num_classes=2048)

        preprocessing_transform = simsiam_preprocessing_transform
        inverse_transform = simsiam_inverse_transform
        training_augmentation = simsiam_training_augmentation
    elif model_name == "pixpro":
        if for_lin_probe:
            model = pixpro_get_model(model_weights_path, num_classes=1000)

            # freeze all layers but the last fc
            for name, param in model[1].named_parameters():
                if name not in ["fc.weight", "fc.bias"]:
                    param.requires_grad = False
            # init the fc layer
            model[1].fc.weight.data.normal_(mean=0.0, std=0.01)
            model[1].fc.bias.data.zero_()

        else:
            model = pixpro_get_model(model_weights_path, num_classes=None)

        preprocessing_transform = pixpro_preprocessing_transform
        inverse_transform = pixpro_inverse_transform
        training_augmentation = pixpro_training_augmentation

    elif model_name == "amdim":
        if for_lin_probe:
            model = amdim_get_model(model_weights_path, num_classes=1000)

            # freeze all layers but the last fc
            for name, param in model[1].named_parameters():
                if name not in ["classifier_head.weight", "classifier_head.bias"]:
                    param.requires_grad = False
            # init the fc layer
            model[2].classifier_head.weight.data.normal_(mean=0.0, std=0.01)
            model[2].classifier_head.bias.data.zero_()

        else:
            model = amdim_get_model(model_weights_path, num_classes=None)

        preprocessing_transform = amdim_preprocessing_transform
        inverse_transform = amdim_inverse_transform
        training_augmentation = amdim_training_augmentation

    elif model_name == "mae":
        if for_lin_probe:
            model = mae_get_model(model_weights_path, num_classes=1000)

            # freeze all but the head
            for _, p in model[1].named_parameters():
                p.requires_grad = False
            for _, p in model[1].head.named_parameters():
                p.requires_grad = True

            # init the head layer
            model[1].head[1].weight.data.normal_(mean=0.0, std=0.01)
            model[1].head[1].bias.data.zero_()

            print("Trainable params:")
            for name, p in model[1].named_parameters():
                if p.requires_grad:
                    print(name)
        else:
            model = mae_get_model(model_weights_path, num_classes=None)

        preprocessing_transform = mae_preprocessing_transform
        inverse_transform = mae_inverse_transform
        training_augmentation = mae_training_augmentation

    elif model_name in ["simclr2_r50_1x_sk0", "simclr2_r50_1x_sk1"]:
        model = simCLR_get_model(model_weights_path)

        if for_lin_probe:
            # freeze all layers but the last fc
            for name, param in model.named_parameters():
                if name not in ["fc.weight", "fc.bias"]:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                # print(f"Param: {name}: req grad = {param.requires_grad}; leaf node = {param.is_leaf}")
            # init the fc layer
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()

        preprocessing_transform = simCLR_preprocessing_transform
        inverse_transform = simCLR_inverse_transform
        training_augmentation = simCLR_training_augmentation

    elif model_name == "resnet":
        if for_lin_probe:
            model = resnet_get_model(model_weights_path, num_classes=1000)

            # freeze all layers but the last fc
            for name, param in model[1].named_parameters():
                if name not in ["fc.weight", "fc.bias"]:
                    param.requires_grad = False
            # init the fc layer
            if resnet_fc_init:
                model[1].fc.weight.data.normal_(mean=0.0, std=0.01)
                model[1].fc.bias.data.zero_()

        else:
            model = resnet_get_model(model_weights_path, num_classes=None)

        preprocessing_transform = resnet_preprocessing_transform
        inverse_transform = resnet_inverse_transform
        training_augmentation = resnet_training_augmentation
    else:
        raise NotImplementedError(f"Model name {model_name} is not recognized")

    return model, preprocessing_transform, inverse_transform, training_augmentation
