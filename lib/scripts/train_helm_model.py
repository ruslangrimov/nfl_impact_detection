import cv2

import sys
import os
import argparse

import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A

from tqdm.auto import tqdm

import importlib

from segmentation_models_pytorch.encoders import get_preprocessing_fn
import segmentation_models_pytorch as smp

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from lib.utils import (
    get_print_fn, imread, get_test_train_split, get_p_mat,
    get_acc
)

from lib.paths import DATA_PATH, PREPARED_PATH, CHECKPOINTS_PATH


class FragmentsDataset(Dataset):
    def __init__(
            self, fragments_path, fragments, preprocess_input,
            img_sz=224, transforms=None, return_grid=True):
        self.fragments_path = fragments_path
        self.fragments = fragments
        self.preprocess_input = preprocess_input
        self.img_sz = img_sz
        self.transforms = transforms
        self.return_grid = return_grid

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx):
        f_path = os.path.join(os.path.join(self.fragments_path, 'neg'), self.fragments[idx])
        if not os.path.isdir(f_path):
            f_path = os.path.join(os.path.join(self.fragments_path, 'pos'), self.fragments[idx])

        i = 4
        x = imread(os.path.join(f_path, f'x_{i}.jpg'))
        mask = imread(os.path.join(f_path, f'y_{i}.png'))
        boxes = np.load(os.path.join(f_path, 'boxes.npy'))

        boxes = (boxes[boxes[:, 0] == 4][:, 1:]).astype(np.int)
        labels = boxes[:, 2]
        boxes = np.concatenate([boxes[:, :2], boxes[:, -2:]], axis=-1)

        pascal_boxes = np.stack([
            boxes[:, 1] - boxes[:, 3] // 2,
            boxes[:, 0] - boxes[:, 2] // 2,
            boxes[:, 1] + boxes[:, 3] // 2,
            boxes[:, 0] + boxes[:, 2] // 2,], axis=-1).clip(0, 224)

        if self.transforms:
            x, mask, pascal_boxes, labels = self.transforms(x, mask, pascal_boxes, labels)
            labels = np.array(labels)
            pascal_boxes = np.array(pascal_boxes)

        if len(pascal_boxes) > 0:
            boxes = pascal_boxes[:, [1, 0, 3, 2]] # xy to yx
        else:
            boxes = np.empty((0, 4), dtype=np.float32)

        assert len(boxes) == len(labels), "WTF?"

        x = self.preprocess_input(x).astype(np.float32).transpose(2, 0, 1)

        if len(boxes) > 0:
            cy = ((boxes[:, 0] + boxes[:, 2]) / 2).round().astype(np.int)
            cx = ((boxes[:, 1] + boxes[:, 3]) / 2).round().astype(np.int)

            r = (((boxes[:, 2] - boxes[:, 0]) + (boxes[:, 3] - boxes[:, 1])) // 4).round().astype(np.int)

            wshs = np.zeros((r.shape[0], self.img_sz, self.img_sz), dtype=np.float32)
            for n in range(wshs.shape[0]):
                tmp = np.zeros((self.img_sz, self.img_sz), dtype=np.uint8)
                tmp = cv2.circle(tmp, (cx[n], cy[n]), r[n], 1, -1)
                tmp = cv2.distanceTransform(tmp, cv2.DIST_L2, 0).astype(np.float32)
                if tmp.max() > 0:
                    wshs[n, ...] = tmp / tmp.max()

            wsh = wshs.max(0, keepdims=True)
        else:
            wsh = np.zeros((1, self.img_sz, self.img_sz), dtype=np.float32)

        return x, wsh


def main(cfg, exp_name, fold=0, num_workers=0):
    _train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.ColorJitter(p=1),
                A.HueSaturationValue(40, 60, 40, p=1),
                A.RGBShift(60, 60, 60, p=1),
                A.RandomGamma((20, 200), p=1)
            ], p=0.8),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0,
                            border_mode=0, p=0.4),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        ),
        p=1.0)

    def train_transforms(img, mask, bboxes, labels):
        out = _train_transforms(image=img, mask=mask, bboxes=bboxes, labels=labels)

        return out['image'], out['mask'], out['bboxes'], out['labels']

    def is_train_f(f):
        return int(f.split('_')[0]) in train_game_ids

    print_l = get_print_fn(exp_name)
    print_l(f"Start {exp_name}")

    batch_size = cfg.batch_size
    non_blocking = False

    device = torch.device('cuda')  # Train on all available videocards

    preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')

    train_labels = pd.read_csv(os.path.join(DATA_PATH, 'train_labels.csv'))
    videos = train_labels['video'].unique()
    game_ids = np.unique([int(v.split('.')[0].split('_')[0]) for v in videos])

    test_game_ids, train_game_ids = get_test_train_split(game_ids, fold=fold)

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
        FragmentsDataset(all_fragments_path, test_fragments, preprocess_input, img_sz=224),
        batch_size, shuffle=False, num_workers=num_workers, pin_memory=non_blocking)

    model = nn.DataParallel(smp.FPN(
        encoder_name=cfg.base_model,
        encoder_depth=5,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None,
        upsampling=4,
    )).cuda()

    lr = cfg.init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))

    epochs = cfg.epochs

    def _get_p_mat(pred_y, y):
        return get_p_mat((pred_y > 0.5).float(), y > 0.5)

    for epoch in tqdm(range(0, epochs)):
        if epoch in cfg.lr_update_epochs:
            lr *= 0.1
            print_l(f"Updated lr to {lr}")
            optimizer = optim.Adam(model.parameters(), lr=lr)

        train_fragments = train_pos_fragments + random.sample(train_neg_all_fragments, 2*len(train_pos_fragments))

        train_loader = DataLoader(
            FragmentsDataset(all_fragments_path, train_fragments, preprocess_input,
                             img_sz=224, transforms=train_transforms),
            batch_size, shuffle=True, num_workers=num_workers, pin_memory=non_blocking
        )

        model.train()

        m = 0
        train_cnt = 0
        train_loss = 0
        for x, wsh in tqdm(train_loader):
            x, wsh = x.to(device), wsh.to(device)
            optimizer.zero_grad()
            pred_y = model(x)
            loss = criterion(pred_y, wsh)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            m = _get_p_mat(pred_y, wsh) + m
            train_cnt += x.shape[0]

        train_loss /= train_cnt
        train_acc = get_acc(m[0], m[1], m[2], m[3])
        train_m = m

        model.eval()

        m = 0
        test_cnt = 0
        test_loss = 0
        for x, wsh in tqdm(test_loader):
            x, wsh = x.to(device), wsh.to(device)
            with torch.no_grad():
                pred_y = model(x)
            loss = criterion(pred_y, wsh)

            test_loss += loss.item()
            m = _get_p_mat(pred_y, wsh) + m
            test_cnt += x.shape[0]

        test_loss /= test_cnt
        test_acc = get_acc(m[0], m[1], m[2], m[3])
        test_m = m

        print_l(
            f"epoch: {epoch+1}, "+
            f"train loss: {train_loss:.6f}, "+
            f"acc: {train_acc[0]:.4f}/{train_acc[1]:.4f}, {int(train_m[0])}/{int(train_m[1])}/{int(train_m[2])}/{int(train_m[3])}, "+
            f"test loss {test_loss:.6f}, "+
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
