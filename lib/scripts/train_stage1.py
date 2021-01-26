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
    get_print_fn, get_test_train_split, get_p_mat, get_f1,
    get_acc
)
from lib.losses.losses import DetectionLoss
from lib.datasets.datasets import get_fragments_dataset
from lib.datasets import augmentations
from lib.models import models

from lib.paths import DATA_PATH, PREPARED_PATH, CHECKPOINTS_PATH


def main(cfg, exp_name, fold=0, num_workers=0):
    def is_train_f(f):
        return int(f.split('_')[0]) in train_game_ids

    print_l = get_print_fn(exp_name)
    print_l(f"Start {exp_name}")

    non_blocking = False
    cnt = cfg.cnt
    batch_size = cfg.batch_size
    img_sz = cfg.img_sz

    device = torch.device('cuda')  # Train on all available videocards

    print_l(f"Creating model {cfg.base_model}")
    model = getattr(models, 'get_'+cfg.base_model)(device)
    model = nn.DataParallel(model)
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

    transforms_fn = getattr(augmentations, 'get_'+cfg.transforms)

    all_fragments_path = os.path.join(PREPARED_PATH, cfg.data_folder)
    pos_fragments_path = os.path.join(all_fragments_path, 'pos')
    neg_fragments_path = os.path.join(all_fragments_path, 'neg')

    pos_fragments = os.listdir(pos_fragments_path)
    neg_fragments = os.listdir(neg_fragments_path)

    train_pos_fragments = [f for f in pos_fragments if is_train_f(f)]
    train_neg_all_fragments = [f for f in neg_fragments if is_train_f(f)]

    test_pos_fragments = [f for f in pos_fragments if not is_train_f(f)]
    test_neg_all_fragments = [f for f in neg_fragments if not is_train_f(f)]

    del pos_fragments, neg_fragments  # to not be confused

    print_l(f"train_pos_fragments: {len(train_pos_fragments)}, train_neg_all_fragments: {len(train_neg_all_fragments)}")
    print_l(f"test_pos_fragments: {len(test_pos_fragments)}, test_neg_all_fragments: {len(test_neg_all_fragments)}")

    random.seed(42)
    test_fragments = test_pos_fragments + random.sample(test_neg_all_fragments, 2*len(test_pos_fragments))

    test_loader = DataLoader(
        get_fragments_dataset(all_fragments_path, test_fragments, cnt=cnt, img_sz=img_sz, transforms=None),
        batch_size, shuffle=False, num_workers=num_workers, pin_memory=non_blocking
    )

    epochs = cfg.epochs

    def _get_p_mat(pred_y, y):
        gt_mask = (y[:, 0] > 0.5)
        pred_y[:, -1][gt_mask]
        return get_p_mat(pred_y[:, -1][gt_mask], y[:, -1][gt_mask])

    for epoch in tqdm(range(0, epochs)):
        if epoch in cfg.lr_update_epochs:
            lr *= 0.1
            print_l(f"Updated lr to {lr}")
            optimizer = optim.Adam(model.parameters(), lr=lr)

        # Hard mining
        if cfg.hard_mining and epoch > 0:
            train_neg_hard_cand_fragments = random.sample(train_neg_all_fragments,
                                                        k=len(train_neg_all_fragments) // 2)

            train_neg_hard_loader = DataLoader(
                get_fragments_dataset(all_fragments_path, train_neg_hard_cand_fragments, cnt=cnt, img_sz=img_sz),
                batch_size, shuffle=False, num_workers=num_workers, pin_memory=non_blocking)

            model.eval()

            neg_lbls = []
            neg_pred_lbls = []
            neg_pred_segm_maxs = []

            for x, lbl, y in tqdm(train_neg_hard_loader):
                x = x.to(device)
                with torch.no_grad():
                    pred_lbl, pred_y = model(x)

                neg_pred_segm_maxs.append(pred_y[:, -1].sigmoid().cpu().view(x.shape[0], -1).max(-1)[0].numpy().tolist())
                neg_pred_lbls.append(pred_lbl[:, 0].sigmoid().cpu().numpy().tolist())
                neg_lbls.append(lbl.cpu()[:, 0].numpy().tolist())

            neg_pred_segm_maxs = np.concatenate(neg_pred_segm_maxs)
            neg_pred_lbls = np.concatenate(neg_pred_lbls)
            neg_lbls = np.concatenate(neg_lbls)

            # Update train dataset with new hard examples
            train_neg_hard_fragments = np.array(train_neg_hard_cand_fragments)[
                ((neg_pred_lbls > cfg.hard_lbl_th) | (neg_pred_segm_maxs > cfg.hard_segm_th)) & (neg_lbls < 0.5)
            ].tolist()

            print_l(f"chose new {len(train_neg_hard_fragments)} hard examples")

            if len(train_neg_hard_fragments) >= 2*len(train_pos_fragments):
                print_l(f"sample {2*len(train_pos_fragments)} hard examples")
                train_fragments = (train_pos_fragments +
                                random.sample(train_neg_hard_fragments, 2*len(train_pos_fragments))
                                )
            else:
                train_fragments = (train_pos_fragments +
                                train_neg_hard_fragments +
                                random.sample(train_neg_all_fragments,
                                                2*len(train_pos_fragments)-len(train_neg_hard_fragments))
                                )
        else:
            train_fragments = train_pos_fragments + random.sample(train_neg_all_fragments, 2*len(train_pos_fragments))

        train_loader = DataLoader(
            get_fragments_dataset(all_fragments_path, train_fragments, cnt=cnt, img_sz=img_sz,
                                transforms=transforms_fn(frames=cnt)),
            batch_size, shuffle=True, num_workers=num_workers, pin_memory=non_blocking)


        model.train()

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
