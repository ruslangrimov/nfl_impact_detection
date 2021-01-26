import sys
import os
import subprocess
import pickle

import argparse

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import importlib

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from lib.utils import get_print_fn
from lib.paths import (
    LIBRARY_PATH, SUBMISSIONS_PATH, TEST_DATA_PATH,
    TMP_PATH, MODELS_PATH
)


def main(cfg):
    exp_name = 'test_inference'

    print_l = get_print_fn(exp_name)
    print_l(f"Start {exp_name}")

    cfg_path = os.path.join(TMP_PATH, 'config.pkl')
    with open(cfg_path, 'wb+') as f:
        pickle.dump(cfg, f)

    videos = sorted([v for v in os.listdir(TEST_DATA_PATH) if v.endswith('.mp4')])
    print_l(f"Total number of videos: {len(videos)}")

    all_pred_bboxes = [pd.DataFrame([],
        columns=['gameKey', 'playID', 'view', 'video',
                'frame', 'left', 'top', 'right', 'bottom',
                'bb_score', 'pos_score'],
    ), ]

    os.makedirs(TMP_PATH, exist_ok=True)

    for video_name in tqdm(videos):
        base_name = video_name.split('.')[0]

        subprocess.run([
                sys.executable,
                os.path.join(LIBRARY_PATH, 'lib/scripts/inference_v0.py'),
                "-cfg", cfg_path,
                "-i", os.path.join(TEST_DATA_PATH, video_name),
                "-o", os.path.join(TMP_PATH, f'{base_name}.csv'),
            ],
            capture_output=False,
            timeout=None,
            check=True
        )

        pred_bboxes = pd.read_csv(os.path.join(TMP_PATH, f'{base_name}.csv'))
        all_pred_bboxes.append(pred_bboxes)

    all_pred_bboxes = pd.concat(all_pred_bboxes, axis=0, ignore_index=True)

    game_ids = set(['_'.join(v.split('_')[:2]) for v in all_pred_bboxes['video']])

    fix_th = cfg.fix_th
    fix_dist = cfg.fix_dist

    all_pred_bboxes_fixed = []

    for game_id in game_ids:
        v0_bboxes_df = all_pred_bboxes.query(f"video == '{game_id}_Sideline.mp4'")
        v1_bboxes_df = all_pred_bboxes.query(f"video == '{game_id}_Endzone.mp4'")

        v0_bboxes = v0_bboxes_df.iloc[:, -7:].values
        v1_bboxes = v1_bboxes_df.iloc[:, -7:].values

        v0 = np.unique(v0_bboxes[:, 0])

        v1 = np.unique(v1_bboxes[:, 0])

        vv = np.abs(v0[:, None] - v1[None, :])

        v0_min_dist = vv.min(axis=1)
        v1_min_dist = vv.min(axis=0)

        v0_bboxes_fixed_df = v0_bboxes_df.iloc[
            np.isin(v0_bboxes[:, 0], v0[v0_min_dist <= fix_dist]) | (v0_bboxes[:, -1] > fix_th)].copy()
        v1_bboxes_fixed_df = v1_bboxes_df.iloc[
            np.isin(v1_bboxes[:, 0], v1[v1_min_dist <= fix_dist]) | (v1_bboxes[:, -1] > fix_th)].copy()

        all_pred_bboxes_fixed.append(v0_bboxes_fixed_df)
        all_pred_bboxes_fixed.append(v1_bboxes_fixed_df)

    all_pred_bboxes_fixed = pd.concat(all_pred_bboxes_fixed, axis=0, ignore_index=True)

    all_pred_bboxes = all_pred_bboxes_fixed

    all_pred_bboxes['width'] = all_pred_bboxes['right'] - all_pred_bboxes['left']
    all_pred_bboxes['height'] = all_pred_bboxes['bottom'] - all_pred_bboxes['top']
    all_pred_bboxes = all_pred_bboxes[["gameKey","playID","view","video","frame","left","width","top","height"]]

    all_pred_bboxes.to_csv(os.path.join(SUBMISSIONS_PATH, 'submission.csv'), index=False)

    print_l(f"Finish {exp_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", dest="cfg",
                        help="config file", required=True, type=str)

    args = parser.parse_args()

    sys.path.append(os.path.dirname(args.cfg))
    cfg = importlib.import_module(
        os.path.splitext(os.path.basename(args.cfg))[0]).cfg

    k_to_fix = ['base_weigths', 'det2_weigths', 'det3_weigths', 'adv_pos_weights']
    for k in k_to_fix:
        if hasattr(cfg, k) and getattr(cfg, k) is not None:
            setattr(cfg, k, [os.path.join(MODELS_PATH, p) for p in getattr(cfg, k)])

    main(cfg)
