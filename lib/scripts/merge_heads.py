import sys
import os

import torch
import torch.nn as nn

import argparse

from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from lib.models.models import (
    get_i3dfpn_bbox_v0, get_i3d3fpn_bbox_v0,
    get_mh_i3dfpn_bbox_v0, get_mh_i3d3fpn_bbox_v0
)
from lib.paths import CHECKPOINTS_PATH, MODELS_PATH


def load_main_model(w, model_fn, device):
    tmp_model = nn.Sequential(OrderedDict([('module', model_fn(device))]))
    tmp_model.load_state_dict(torch.load(w))
    model = tmp_model.module

    return model


def main(input_paths, output_path):
    device = torch.device('cpu')
    weights_path = os.path.join(CHECKPOINTS_PATH, input_paths[0])
    tmp = torch.load(weights_path, map_location=device)
    if tmp['module.model.s1.pathway0_stem.conv.weight'].shape[-3] == 3:
        model_fn = get_i3d3fpn_bbox_v0
        mh_model_fn = get_mh_i3d3fpn_bbox_v0
    else:
        model_fn = get_i3dfpn_bbox_v0
        mh_model_fn = get_mh_i3dfpn_bbox_v0

    del tmp

    mh_model = mh_model_fn(device, len(input_paths))

    for n, w in enumerate(input_paths):
        w = os.path.join(CHECKPOINTS_PATH, w)
        tmp_model = load_main_model(w, model_fn, device)

        if n == 0:
            mh_model.model.load_state_dict(tmp_model.state_dict())

        mh_model.grid_heads[n].load_state_dict(tmp_model.grid_head.state_dict())
        mh_model.fpn_decoders[n].load_state_dict(tmp_model.fpn_decoder.state_dict())

        del tmp_model

    torch.save(mh_model.state_dict(), os.path.join(MODELS_PATH, output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="input_paths",
                        help="input model paths", required=True, type=str, nargs='+')
    parser.add_argument("-o", dest="output_path",
                        help="output model path", required=True, type=str)
    args = parser.parse_args()

    main(args.input_paths, args.output_path)
