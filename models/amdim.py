import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageFilter
import numpy as np

# Code repurposed from https://github.com/Philip-Bachman/amdim-public
MP_CONFIG = {"enabled": False, "optimization_level": "O2"}


def has_many_gpus():
    return torch.cuda.device_count() >= 6


class PickOnlyFirstFlatten(nn.Module):
    def __init__(self):
        super(PickOnlyFirstFlatten, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=(7, 8))

    def forward(self, x):
        r = self.avg_pool(x[0]).flatten(start_dim=1)
        return r


class PickOnlyFirstClassifier(nn.Module):
    def __init__(self):
        super(PickOnlyFirstClassifier, self).__init__()

        self.avg_pool = nn.AvgPool2d(kernel_size=(7, 8))
        self.classifier_head = nn.Linear(2048, 1000)

    def forward(self, x):
        r = self.classifier_head(self.avg_pool(x[0]).flatten(start_dim=1))
        return r


class Encoder(nn.Module):
    def __init__(
        self,
        dummy_batch,
        num_channels=3,
        ndf=64,
        n_rkhs=512,
        n_depth=3,
        encoder_size=32,
        use_bn=False,
    ):
        super(Encoder, self).__init__()
        self.ndf = ndf
        self.n_rkhs = n_rkhs
        self.use_bn = use_bn
        self.dim2layer = None

        # encoding block for local features
        print("Using a {}x{} encoder".format(encoder_size, encoder_size))
        if encoder_size == 32:
            self.layer_list = nn.ModuleList(
                [
                    Conv3x3(num_channels, ndf, 3, 1, 0, False),
                    ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),
                    ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 2, ndf * 4, 2, 2, 0, n_depth, use_bn),
                    MaybeBatchNorm2d(ndf * 4, True, use_bn),
                    ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                    ConvResNxN(ndf * 4, n_rkhs, 3, 1, 0, use_bn),
                    MaybeBatchNorm2d(n_rkhs, True, True),
                ]
            )
        elif encoder_size == 64:
            self.layer_list = nn.ModuleList(
                [
                    Conv3x3(num_channels, ndf, 3, 1, 0, False),
                    ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                    MaybeBatchNorm2d(ndf * 8, True, use_bn),
                    ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                    ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                    MaybeBatchNorm2d(n_rkhs, True, True),
                ]
            )
        elif encoder_size == 128:
            self.layer_list = nn.ModuleList(
                [
                    Conv3x3(num_channels, ndf, 5, 2, 2, False, pad_mode="reflect"),
                    Conv3x3(ndf, ndf, 3, 1, 0, False),
                    ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                    MaybeBatchNorm2d(ndf * 8, True, use_bn),
                    ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                    ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                    ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                    MaybeBatchNorm2d(n_rkhs, True, True),
                ]
            )
        else:
            raise RuntimeError("Could not build encoder." "Encoder size {} is not supported".format(encoder_size))
        self._config_modules(dummy_batch, [1, 5, 7], n_rkhs, use_bn)

    def init_weights(self, init_scale=1.0):
        """
        Run custom weight init for modules...
        """
        for layer in self.layer_list:
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
        for layer in self.modules():
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
            if isinstance(layer, FakeRKHSConvNet):
                layer.init_weights(init_scale)

    def _config_modules(self, x, rkhs_layers, n_rkhs, use_bn):
        """
        Configure the modules for extracting fake rkhs embeddings for infomax.
        """
        enc_acts = self._forward_acts(x)
        self.dim2layer = {}
        for i, h_i in enumerate(enc_acts):
            for d in rkhs_layers:
                if h_i.size(2) == d:
                    self.dim2layer[d] = i
        # get activations and feature sizes at different layers
        self.ndf_1 = enc_acts[self.dim2layer[1]].size(1)
        self.ndf_5 = enc_acts[self.dim2layer[5]].size(1)
        self.ndf_7 = enc_acts[self.dim2layer[7]].size(1)
        # configure modules for fake rkhs embeddings
        self.rkhs_block_1 = NopNet()
        self.rkhs_block_5 = FakeRKHSConvNet(self.ndf_5, n_rkhs, use_bn)
        self.rkhs_block_7 = FakeRKHSConvNet(self.ndf_7, n_rkhs, use_bn)

    def _forward_acts(self, x):
        """
        Return activations from all layers.
        """
        # run forward pass through all layers
        layer_acts = [x]
        for _, layer in enumerate(self.layer_list):
            layer_in = layer_acts[-1]
            layer_out = layer(layer_in)
            layer_acts.append(layer_out)
        # remove input from the returned list of activations
        return_acts = layer_acts[1:]
        return return_acts

    def forward(self, x):
        """
        Compute activations and Fake RKHS embeddings for the batch.
        """
        if has_many_gpus():
            if x.abs().mean() < 1e-4:
                r1 = torch.zeros((1, self.n_rkhs, 1, 1), device=x.device, dtype=x.dtype).detach()
                r5 = torch.zeros((1, self.n_rkhs, 5, 5), device=x.device, dtype=x.dtype).detach()
                r7 = torch.zeros((1, self.n_rkhs, 7, 7), device=x.device, dtype=x.dtype).detach()
                return r1, r5, r7
        # compute activations in all layers for x
        acts = self._forward_acts(x)
        # gather rkhs embeddings from certain layers
        r1 = self.rkhs_block_1(acts[self.dim2layer[1]])
        r5 = self.rkhs_block_5(acts[self.dim2layer[5]])
        r7 = self.rkhs_block_7(acts[self.dim2layer[7]])
        return r1, r5, r7


class MaybeBatchNorm2d(nn.Module):
    def __init__(self, n_ftr, affine, use_bn):
        super(MaybeBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(n_ftr, affine=affine)
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
        return x


class NopNet(nn.Module):
    def __init__(self, norm_dim=None):
        super(NopNet, self).__init__()
        self.norm_dim = norm_dim

    def forward(self, x):
        if self.norm_dim is not None:
            x_norms = torch.sum(x**2.0, dim=self.norm_dim, keepdim=True)
            x_norms = torch.sqrt(x_norms + 1e-6)
            x = x / x_norms
        return x


class Conv3x3(nn.Module):
    def __init__(self, n_in, n_out, n_kern, n_stride, n_pad, use_bn=True, pad_mode="constant"):
        super(Conv3x3, self).__init__()
        assert pad_mode in ["constant", "reflect"]
        self.n_pad = (n_pad, n_pad, n_pad, n_pad)
        self.pad_mode = pad_mode
        self.conv = nn.Conv2d(n_in, n_out, n_kern, n_stride, 0, bias=(not use_bn))
        self.relu = nn.ReLU(inplace=True)
        self.bn = MaybeBatchNorm2d(n_out, True, use_bn) if use_bn else None

    def forward(self, x):
        if self.n_pad[0] > 0:
            # pad the input if required
            x = F.pad(x, self.n_pad, mode=self.pad_mode)
        # conv is always applied
        x = self.conv(x)
        # apply batchnorm if required
        if self.bn is not None:
            x = self.bn(x)
        # relu is always applied
        out = self.relu(x)
        return out


class MLPClassifier(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super(MLPClassifier, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(Flatten(), nn.Dropout(p=p), nn.Linear(n_input, n_classes, bias=True))
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True),
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class FakeRKHSConvNet(nn.Module):
    def __init__(self, n_input, n_output, use_bn=False):
        super(FakeRKHSConvNet, self).__init__()
        self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_output, n_output, kernel_size=1, stride=1, padding=0, bias=False)
        # BN is optional for hidden layer and always for output layer
        self.bn_hid = MaybeBatchNorm2d(n_output, True, use_bn)
        self.bn_out = MaybeBatchNorm2d(n_output, True, True)
        self.shortcut = nn.Conv2d(n_input, n_output, kernel_size=1, stride=1, padding=0, bias=True)
        # initialize shortcut to be like identity (if possible)
        if n_output >= n_input:
            eye_mask = np.zeros((n_output, n_input, 1, 1), dtype=bool)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = True
            self.shortcut.weight.data.uniform_(-0.01, 0.01)
            self.shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.0)
        return

    def init_weights(self, init_scale=1.0):
        # initialize first conv in res branch
        # -- rescale the default init for nn.Conv2d layers
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        self.conv1.weight.data.mul_(init_scale)
        # initialize second conv in res branch
        # -- set to 0, like fixup/zero init
        nn.init.constant_(self.conv2.weight, 0.0)
        return

    def forward(self, x):
        h_res = self.conv2(self.relu1(self.bn_hid(self.conv1(x))))
        h = self.bn_out(h_res + self.shortcut(x))
        return h


class ConvResNxN(nn.Module):
    def __init__(self, n_in, n_out, width, stride, pad, use_bn=False):
        super(ConvResNxN, self).__init__()
        assert n_out >= n_in
        self.n_in = n_in
        self.n_out = n_out
        self.width = width
        self.stride = stride
        self.pad = pad
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_in, n_out, width, stride, pad, bias=False)
        self.conv2 = nn.Conv2d(n_out, n_out, 1, 1, 0, bias=False)
        self.conv3 = None
        # ...
        self.bn1 = MaybeBatchNorm2d(n_out, True, use_bn)
        return

    def init_weights(self, init_scale=1.0):
        # initialize first conv in res branch
        # -- rescale the default init for nn.Conv2d layers
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        self.conv1.weight.data.mul_(init_scale)
        # initialize second conv in res branch
        # -- set to 0, like fixup/zero init
        nn.init.constant_(self.conv2.weight, 0.0)
        return

    def forward(self, x):
        h1 = self.bn1(self.conv1(x))
        h2 = self.conv2(self.relu2(h1))
        if self.n_out < self.n_in:
            h3 = self.conv3(x)
        elif self.n_in == self.n_out:
            h3 = F.avg_pool2d(x, self.width, self.stride, self.pad)
        else:
            h3_pool = F.avg_pool2d(x, self.width, self.stride, self.pad)
            h3 = F.pad(h3_pool, (0, 0, 0, 0, 0, self.n_out - self.n_in))
        h23 = h2 + h3
        return h23


class ConvResBlock(nn.Module):
    def __init__(self, n_in, n_out, width, stride, pad, depth, use_bn):
        super(ConvResBlock, self).__init__()
        layer_list = [ConvResNxN(n_in, n_out, width, stride, pad, use_bn)]
        for i in range(depth - 1):
            layer_list.append(ConvResNxN(n_out, n_out, 1, 1, 0, use_bn))
        self.layer_list = nn.Sequential(*layer_list)
        return

    def init_weights(self, init_scale=1.0):
        """
        Do a fixup-style init for each ConvResNxN in this block.
        """
        for m in self.layer_list:
            m.init_weights(init_scale)
        return

    def forward(self, x):
        # run forward pass through the list of ConvResNxN layers
        x_out = self.layer_list(x)
        return x_out


def amdim_get_model(weigths_path, num_classes=None):

    ckp = torch.load(weigths_path)
    hp = ckp["hyperparams"]
    params = ckp["model"]
    dummy_batch = torch.zeros((2, 3, hp["encoder_size"], hp["encoder_size"]))
    model = Encoder(
        dummy_batch=dummy_batch,
        ndf=hp["ndf"],
        n_rkhs=hp["n_rkhs"],
        n_depth=hp["n_depth"],
        encoder_size=hp["encoder_size"],
    )
    # params = {k.replace('encoder.module.', 'encoder.'): v for k, v in params.items()}
    params = {
        k.replace("encoder.module.", ""): v
        for k, v in params.items()
        if not k.startswith("evaluator") and not k == "g2l_loss.masks_r5"
    }
    model.load_state_dict(params)

    steps = [
        torchvision.transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model,
    ]

    # add linear classifier if required
    if num_classes is not None:
        steps.append(PickOnlyFirstClassifier())
    else:
        steps.append(PickOnlyFirstFlatten())

    normalized_model = torch.nn.Sequential(*steps)

    return normalized_model


amdim_preprocessing_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 244)),
        torchvision.transforms.ToTensor(),
    ]
)

amdim_inverse_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
    ]
)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


amdim_training_augmentation = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop((224, 244), scale=(0.2, 1.0)),
        torchvision.transforms.RandomApply(
            [torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],  # not strengthened
            p=0.8,
        ),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ]
)
