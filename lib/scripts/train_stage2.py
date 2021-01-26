import sys
import os
import argparse

import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import importlib

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from lib.utils import (
    get_print_fn, get_test_train_split, load_any_base_model, get_p_mat, get_f1,
    get_acc
)
from lib.losses.losses import DetectionLoss
from lib.datasets.datasets import get_wholeimage_dataset
from lib.datasets import augmentations

from lib.paths import DATA_PATH, WHOLE_IMG_PATH, CHECKPOINTS_PATH


def main(cfg, exp_name, fold=0, num_workers=0):
    print_l = get_print_fn(exp_name)
    print_l(f"Start {exp_name}")

    non_blocking = False
    cnt = cfg.cnt
    batch_size = cfg.batch_size

    device = torch.device('cuda')  # Train on all available videocards

    print_l(f"Loading model {cfg.base_model} with weights {cfg.model_weights}")
    model = load_any_base_model(cfg.model_weights, device)
    model = nn.DataParallel(model)

    for _, p in model.module.model.named_parameters():
        p.requires_grad = False

    model.module.model.eval()
    model.cuda()

    lr = cfg.init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = DetectionLoss(
        box_weight=cfg.box_weight, pos_weight=cfg.pos_weight,
        loss_weights=cfg.loss_weights, class_mask=cfg.class_mask).to(device)
    lbl_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device=device))

    train_labels = pd.read_csv(os.path.join(DATA_PATH, 'train_labels.csv'))
    videos = train_labels['video'].unique()
    game_ids = np.unique([int(v.split('.')[0].split('_')[0]) for v in videos])

    test_game_ids, train_game_ids = get_test_train_split(game_ids, fold=fold)

    print_l(f"Number of impacts {train_labels['impact'].sum()}")

    if cfg.exp_frames is not None:
        imp_train_labels = train_labels[train_labels['impact'] > 0].copy()
        for _, row in tqdm(imp_train_labels.iterrows(), total=len(imp_train_labels)):
            frames = np.array(cfg.exp_frames) + row.frame
            train_labels.loc[(train_labels['video'] == row.video) &
                            (train_labels['frame'].isin(frames)) &
                            (train_labels['label'] == row.label), 'impact'] = 1

        print_l(f"Number of impacts after expansion {train_labels['impact'].sum()}")

    train_train_labels = train_labels[train_labels['gameKey'].isin(train_game_ids)].copy()
    test_train_labels = train_labels[train_labels['gameKey'].isin(test_game_ids)].copy()

    transforms_fn = getattr(augmentations, 'get_'+cfg.transforms)

    test_loader = DataLoader(
        get_wholeimage_dataset(
            WHOLE_IMG_PATH, test_train_labels, cnt=cnt, transforms=None, return_grid=True, add_all=cfg.add_all),
        batch_size, shuffle=False, num_workers=num_workers, pin_memory=non_blocking
    )

    epochs = cfg.epochs

    def _get_p_mat(pred_y, y):
        gt_mask = (y[:, 0] > 0.5)
        return get_p_mat(pred_y[:, -1][gt_mask], y[:, -1][gt_mask])

    for epoch in tqdm(range(0, epochs)):
        if epoch in cfg.lr_update_epochs:
            lr *= 0.1
            print_l(f"Updated lr to {lr}")
            optimizer = optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(
            get_wholeimage_dataset(
                WHOLE_IMG_PATH, train_train_labels, cnt=cnt,
                transforms=transforms_fn(frames=cnt),
                return_grid=True, add_all=cfg.add_all),
            batch_size, shuffle=True, num_workers=num_workers, pin_memory=non_blocking)

        model.module.model.eval()
        model.module.grid_head.train()
        model.module.lbl_head.train()

        m = 0
        train_cnt = 0
        train_loss = 0
        for x, lbl, y in tqdm(train_loader):
            x, lbl, y = x.to(device), lbl.to(device), y.to(device)
            optimizer.zero_grad()
            pred_lbl, pred_y = model(x)
            loss = lbl_criterion(pred_lbl, lbl) + criterion(pred_y, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            m = _get_p_mat(pred_y, y) + m
            train_cnt += x.shape[0]

        train_loss /= train_cnt
        train_f1 = get_f1(m[0], m[1], m[2], m[3])
        train_acc = get_acc(m[0], m[1], m[2], m[3])
        train_m = m

        model.eval()

        m = 0
        test_cnt = 0
        test_loss = 0
        for x, lbl, y in tqdm(test_loader):
            x, lbl, y = x.to(device), lbl.to(device), y.to(device)
            with torch.no_grad():
                pred_lbl, pred_y = model(x)
            loss = lbl_criterion(pred_lbl, lbl) + criterion(pred_y, y)

            test_loss += loss.item()
            m = _get_p_mat(pred_y, y) + m
            test_cnt += x.shape[0]

            # break

        test_loss /= test_cnt
        test_f1 = get_f1(m[0], m[1], m[2], m[3])
        test_acc = get_acc(m[0], m[1], m[2], m[3])
        test_m = m

        print_l(f"epoch: {epoch+1}, "+
            f"train loss: {train_loss:.4f}, "+
            f"acc: {train_acc[0]:.4f}/{train_acc[1]:.4f}, {int(train_m[0])}/{int(train_m[1])}/{int(train_m[2])}/{int(train_m[3])}, "+
            f"test loss {test_loss:.4f}, "+
            f"acc: {test_acc[0]:.4f}/{test_acc[1]:.4f}, {int(test_m[0])}/{int(test_m[1])}/{int(test_m[2])}/{int(test_m[3])} "
            )

        torch.save(model.state_dict(), os.path.join(CHECKPOINTS_PATH, f'{exp_name}_e{epoch}.pth'))

    print_l(f"Finish {exp_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", dest="cfg",
                        help="config file", required=True, type=str)
    parser.add_argument("-e", dest="exp_name",
                        help="experiment name", required=True, type=str)
    parser.add_argument("-f", dest="fold",
                        help="fold", required=True, type=int)
    parser.add_argument("-w", dest="w",
                        help="number of workers", required=True, type=int)
    args = parser.parse_args()

    sys.path.append(os.path.dirname(args.cfg))
    cfg = importlib.import_module(
        os.path.splitext(os.path.basename(args.cfg))[0])

    main(cfg, args.exp_name, args.fold, args.w)
