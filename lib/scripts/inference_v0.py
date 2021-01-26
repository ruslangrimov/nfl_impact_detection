import cv2
import sys
import pickle
import argparse
import os

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from lib.utils import (
    video2imgs, get_bboxes_v1, get_bboxes_v2,
    get_final_pred_bboxes, get_final_pred_bboxes_v2,
    load_any_base_model)
from lib.datasets.datasets import (
    get_wholeimage_inference_dataset, get_wholeimage512_inference_dataset,
    get_wholerawimage_dataset)

# from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchEval
# from effdet.efficientdet import HeadNet

from segmentation_models_pytorch.encoders import get_preprocessing_fn
import segmentation_models_pytorch as smp


def load_det_model_v2(checkpoint_path, device):
    raise NotImplementedError("det_v2 is detector no longer used")


def load_det_model_v3(checkpoint_path, device):
    model = smp.FPN(
        encoder_name="resnet34",
        encoder_depth=5,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
        upsampling=4,  # 4,
    )
    tmp_model = nn.Sequential(OrderedDict([('module', model)]))
    tmp_model.load_state_dict(torch.load(checkpoint_path))
    model = tmp_model.module
    return model.to(device)


def get_pred_helms(model, device, v_imgs, cnt, batch_size, num_workers,
                   non_blocking, ms=[1,], ws=[1.0,]):
    preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')

    inference_loader = DataLoader(
        get_wholerawimage_dataset(v_imgs, preprocess_input, pad=False),
        batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=non_blocking
    )

    model.eval()

    helm_scores = []
    for imgs in inference_loader:
        imgs = imgs.to(device, non_blocking=non_blocking)

        b_helm_scores = []
        for m, w in zip(ms, ws):
            with torch.no_grad():
                m_imgs = F.interpolate(imgs, scale_factor=m, mode='bilinear')
                h_p = (32 - m_imgs.shape[2] % 32) % 32
                if h_p != 0:
                    m_imgs = F.pad(m_imgs, [0, 0, 0, h_p])
                res = model(m_imgs)

                m_imgs = F.interpolate((res[:, :, :-h_p] if h_p != 0 else res).sigmoid(), scale_factor=1/m, mode='bilinear')
                m_imgs = F.pad(m_imgs, [0, 0, 0, 16])
                tmps = m_imgs.cpu().numpy()
                helm_m_score = []
                for tmp in tmps:
                    helm_m_score.append(cv2.resize(tmp[0], None, fx=0.25, fy=0.25,
                                                   interpolation=cv2.INTER_LANCZOS4)[None, ...])
                helm_m_score = np.stack(helm_m_score, axis=0)
            b_helm_scores.append(helm_m_score*w)

        b_helm_scores = np.stack(b_helm_scores, axis=0).sum(0)
        helm_scores.append(b_helm_scores)

    helm_scores = np.concatenate(helm_scores, axis=0)

    return helm_scores


def get_pred_grids(model, device, v_imgs, cnt, batch_size, num_workers,
                   non_blocking):
    inference_loader = DataLoader(
        get_wholeimage_inference_dataset(v_imgs, cnt=cnt),
        batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=non_blocking
    )

    model.eval()

    pred_grids = []
    for imgs in inference_loader:
        imgs = imgs.to(device, non_blocking=non_blocking)

        with torch.no_grad():
            res = model(imgs)
            res = res[1] if isinstance(res, tuple) else res
            pred_grids.append(res.sigmoid().cpu().numpy())

    return np.concatenate(pred_grids, axis=0)


def get_pred_dets(model, device, v_imgs, num_workers, non_blocking):
    data_loader = DataLoader(
        get_wholeimage512_inference_dataset(v_imgs),
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    model.eval()

    dets = []
    for imgs in data_loader:
        imgs = imgs.to(device, non_blocking=non_blocking)

        with torch.no_grad():
            det = model(imgs, torch.tensor([1]*imgs.shape[0]).float().to(device))

        dets.append(det.detach().cpu())

    dets = torch.cat(dets, dim=0)

    return dets


def main(cfg, video_path, csv_path):
    video_name = os.path.basename(video_path)
    base_name = video_name.split('.')[0]

    device = torch.device('cuda:0')

    base_models = []
    for model_weigths in cfg.base_weigths:
        print(f"Loading {os.path.basename(model_weigths)}")
        model = load_any_base_model(model_weigths, device)
        model.eval()
        base_models.append(model)
        del model

    if cfg.bbox2:
        det_model = load_det_model_v2(cfg.det2_weigths, device)
        det_model.eval()

    if cfg.bbox3:
        helm_models = []
        for model_name, model_weigths in zip(cfg.det3_models, cfg.det3_weigths):
            helm_model = load_det_model_v3(model_weigths, device)
            helm_model.eval()
            helm_models.append(helm_model)

    v_imgs = video2imgs(video_path, cfg.scale)
    depth, height, width = v_imgs.shape[:3]

    if cfg.bbox2:
        dets = get_pred_dets(det_model, device, v_imgs, cfg.num_workers,
                             cfg.non_blocking)

    if cfg.bbox3:
        all_helm_scores = []
        for helm_model in helm_models:
            m_all_helm_scores = get_pred_helms(
                helm_model, device, v_imgs, cfg.cnt,
                cfg.batch_size, cfg.num_workers,
                cfg.non_blocking, ms=cfg.bbox3_ms, ws=cfg.bbox3_ws)

            all_helm_scores.append(m_all_helm_scores)

        all_helm_scores = np.stack(all_helm_scores, axis=0).mean(0)

    p_grid = np.zeros((depth-cfg.cnt, 6, int(184*cfg.scale), int(320*cfg.scale)), dtype=np.float32)
    # p_grid = np.zeros((depth-cfg.cnt, 6, 96, int(320*cfg.scale)), dtype=np.float32)
    p_cnt = 0
    for model in base_models:
        p_grids = get_pred_grids(model, device, v_imgs, cfg.cnt,
                                 cfg.batch_size, cfg.num_workers, cfg.non_blocking)
        print(p_grids.shape)
        if len(p_grids.shape) > 4:
            if cfg.MEAN_MH:
                for i in range(p_grids.shape[1]):
                    p_grid += p_grids[:, i] / p_grids.shape[1]
                p_cnt += 1
            else:
                for i in range(p_grids.shape[1]):
                    p_grid += p_grids[:, i]
                    p_cnt += 1
        else:
            p_grid += p_grids
            p_cnt += 1
    p_grid /= p_cnt

    pos_th = cfg._pos_th

    start_p = cfg.cnt // 2
    print(f"start_p {start_p}")

    all_boxes = []
    all_pos_boxes = []
    for f in range(start_p, v_imgs.shape[0]-start_p):
        gf = f-start_p
        if cfg.bbox3:
            helm_scores = all_helm_scores[f, 0]
        else:
            helm_scores = p_grid[gf, 0]

        helm_mask = helm_scores > cfg.helm_th

        pos_map = p_grid[gf, -1]
        if cfg.mult_pos_by_helm_scores:
            pos_map = pos_map * helm_mask # * (1 if lbl_th is None else (p_lbl[gf, 0] > lbl_th))

        if cfg.mult_helm_scores_by_pos_th:
            helm_scores = helm_scores * (pos_map > pos_th)

        if cfg.bbox2:
            b_bb, top, left, bottom, right, b_pos = get_bboxes_v2(
                dets, cfg.helm_th2, f, pos_map, cfg.pos_type, cfg.cell_sz)
        else:
            b_bb, top, left, bottom, right, b_pos = get_bboxes_v1(
                p_grid, gf, helm_mask.astype(np.bool), helm_scores, pos_map,
                cfg.pos_type, cfg.img_sz, cfg.cell_sz)

        b_idxs = torchvision.ops.nms(
            torch.from_numpy(np.stack([left, top, right, bottom], -1)).float(),
            torch.from_numpy(b_pos if cfg.use_pos_max else b_bb),
            iou_threshold=cfg.iou_threshold)

        # for i in range(b_bb.shape[0]):
        for i in b_idxs:
            all_boxes.append((f, b_bb[i], top[i], left[i], bottom[i],
                              right[i], b_pos[i]))
            if b_pos[i] > pos_th:
                all_pos_boxes.append((f, left[i], top[i], right[i],
                                      bottom[i], b_bb[i], b_pos[i]))

    if cfg.final_pred_bboxes_v2:
        print('get_final_pred_bboxes_v2')
        pred_bboxes = get_final_pred_bboxes_v2(
            all_boxes,
            min_iou_threshold=cfg.final_iou_threshold,
            score_field='pos_score' if cfg.final_use_pos_max else 'bb_score',
            score_th=pos_th,
            supress_range=cfg.supress_range)
    else:
        print('get_final_pred_bboxes')
        pred_bboxes = get_final_pred_bboxes(
            all_pos_boxes, final_iou_threshold=cfg.final_iou_threshold,
            score_field='pos_score' if cfg.final_use_pos_max else 'bb_score', supress_range=cfg.supress_range)
    # print(pred_bboxes.shape)

    gameKey, playID, view = video_name.split('.')[0].split('_')
    gameKey, playID = [int(a) for a in [gameKey, playID]]

    pred_bboxes['gameKey'] = gameKey
    pred_bboxes['playID'] = playID
    pred_bboxes['view'] = view
    pred_bboxes['video'] = video_name

    if cfg.scale != 1:
        pred_bboxes['left'] /= cfg.scale
        pred_bboxes['top'] /= cfg.scale
        pred_bboxes['right'] /= cfg.scale
        pred_bboxes['bottom'] /= cfg.scale

    pred_bboxes.to_csv(csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", dest="cfg",
                        help="config pkl", required=True, type=str)
    parser.add_argument("-i", dest="input",
                        help="input video path", required=True, type=str)
    parser.add_argument("-o", dest="output",
                        help="output csv path", required=True, type=str)
    args = parser.parse_args()

    with open(args.cfg, 'rb') as f:
        cfg = pickle.load(f)

    main(cfg, args.input, args.output)
