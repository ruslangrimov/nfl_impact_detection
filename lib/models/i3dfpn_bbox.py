import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from slowfast.models import build_model
from slowfast.config.defaults import get_cfg

from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.base import SegmentationHead


class BlockOutWrapper(nn.Module):
    def __init__(self, model, features):
        super().__init__()
        self.model = model
        self.features = features

    def forward(self, *args):
        out = self.model(*args)
        d = out[0].device.index
        self.features[d].append(out)
        return out


class ModelWrapper(nn.Module):
    def __init__(self, cfg, model_type, device, add_4chan=True, out_channels=6):
        super().__init__()
        self.model = build_model(cfg, 0)
        self.add_4chan = add_4chan
        if self.add_4chan:
            if model_type == 'i3d':
                self.model.s1.pathway0_stem.conv =\
                    nn.Conv3d(4, 64, kernel_size=[5, 7, 7],
                            stride=[1, 2, 2], padding=[2, 3, 3], bias=False)
            elif model_type == 'i3d3':
                self.model.s1.pathway0_stem.conv =\
                    nn.Conv3d(4, 64, kernel_size=[3, 7, 7],
                            stride=[1, 2, 2], padding=[0, 3, 3], bias=False)
            elif model_type == 'c2d':
                self.model.s1.pathway0_stem.conv =\
                    nn.Conv3d(4, 64, kernel_size=[1, 7, 7],
                            stride=[1, 2, 2], padding=[0, 3, 3], bias=False)
            elif model_type[:3] == 'x3d':
                self.model.s1.pathway0_stem.conv_xy =\
                    nn.Conv3d(4, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        else:
            if model_type == 'i3d3':
                self.model.s1.pathway0_stem.conv =\
                    nn.Conv3d(3, 64, kernel_size=[3, 7, 7],
                            stride=[1, 2, 2], padding=[0, 3, 3], bias=False)

        self.model.head = nn.Identity().to(device)
        self.max_devices_cnt = 16
        self.features = {d: [] for d in range(self.max_devices_cnt)}

        self.model.s2 = BlockOutWrapper(self.model.s2, self.features)
        self.model.s3 = BlockOutWrapper(self.model.s3, self.features)
        self.model.s4 = BlockOutWrapper(self.model.s4, self.features)
        self.model.s5 = BlockOutWrapper(self.model.s5, self.features)

        self.fpn_decoder = FPNDecoder(
            encoder_channels=[24, 48, 96, 192] if model_type == 'x3d' else [256, 512, 1024, 2048],
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy='add'
        ).to(device)

        self.grid_head = SegmentationHead(
            in_channels=128,
            out_channels=out_channels,
            kernel_size=3,
            activation=None,
            upsampling=1,
        ).to(device)

        self.lbl_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(192  if model_type == 'x3d' else 2048, 1)
        ).to(device)

    def forward(self, x, return_features=False):
        def _f3dtof2d(f):
            return F.adaptive_avg_pool3d(f, (1,)+f.shape[-2:]).squeeze(-3)

        sh = list(x.shape)
        sh[1] = 1
        if self.add_4chan:
            f_mask = torch.zeros(sh, device=x.device)
            f_mask[:, :, f_mask.shape[2] // 2] = 1.0
            x = torch.cat([x, f_mask], dim=1)
        x = self.model([x, ])
        d = x[0].device.index

        # Predict label
        if not return_features:
            lbl = self.lbl_head(x[0])

            # Predict segmentation
            features = [_f3dtof2d(f[0]) for f in self.features[d]]
            # print([(f.shape, f.device) for f in features])
            grid = self.grid_head(self.fpn_decoder(*features))

        if not return_features:
            self.features[d].clear()  # To clear links to tensors

        return self.features if return_features else (lbl, grid)


def get_model(device, model_type='i3d', add_4chan=True, out_channels=6):
    cfg = get_cfg()
    base_dir = os.path.dirname(__file__)
    if model_type[:3] == 'i3d':
        base_config = './slowfast/configs/Kinetics/I3D_8x8_R50.yaml'
    elif model_type == 'c2d':
        base_config = './slowfast/configs/Kinetics/C2D_8x8_R50.yaml'
    elif model_type == 'x3ds':
        base_config = './slowfast/configs/Kinetics/X3D_S.yaml'
    else:
        raise ValueError(f"Unknown base model type {model_type}")

    cfg.merge_from_file(
        os.path.join(base_dir, base_config))
    cfg.NUM_GPUS = 1
    cfg.DETECTION.ENABLE = False
    cfg.MODEL.ARCH = model_type[:3]  # 'c2d', 'i3d', 'slow', 'x3d'

    model = ModelWrapper(cfg, model_type, device, add_4chan, out_channels)

    return model


class MultiHeadModelWrapper(nn.Module):
    def __init__(self, model, head_number=1):
        super().__init__()
        self.model = model

        self.grid_heads = nn.ModuleList()
        self.fpn_decoders = nn.ModuleList()
        for _ in range(head_number):
            self.grid_heads.append(copy.deepcopy(self.model.grid_head))
            self.fpn_decoders.append(copy.deepcopy(self.model.fpn_decoder))

    def forward(self, x):
        def _f3dtof2d(f):
            return F.adaptive_avg_pool3d(f, (1,)+f.shape[-2:]).squeeze(-3)

        features = self.model(x, return_features=True)
        d = x.device.index

        features = [_f3dtof2d(f[0]) for f in features[d]]

        grids = []
        for grid_head, fpn_decoder in zip(self.grid_heads, self.fpn_decoders):
            grids.append(grid_head(fpn_decoder(*features)))

        self.model.features[d].clear()  # To clear links to tensors

        return torch.stack(grids, dim=1)
