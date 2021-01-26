import cv2
import os
import errno
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict

import torch.nn as nn

import torchvision

from .models import models

def get_print_fn(exp_name):
    from .paths import LOGS_PATH

    def _print_l(s, logs, exp_name):
        logs.append(s)
        print(s)
        os.makedirs(LOGS_PATH, exist_ok=True)
        with open(os.path.join(LOGS_PATH, f"logs_{exp_name}.txt"), 'w+') as f:
            for s in logs:
                f.write(s+"\n")
    logs = []
    print_fn = lambda s: _print_l(s, logs, exp_name)

    return print_fn


def get_test_train_split(game_ids, fold=0):
    if fold == 0:  # Special case for fold 0
        np.random.seed(42)

        game_ids.sort()

        test_game_ids = np.random.choice(game_ids, 10)
        train_game_ids = game_ids[~np.isin(game_ids, test_game_ids)]
    elif fold == -1:  # All examples except a few for test
        np.random.seed(42)

        game_ids.sort()

        test_game_ids = np.random.choice(game_ids, 10)
        train_game_ids = game_ids[~np.isin(game_ids, test_game_ids)]

        train_game_ids, test_game_ids = np.concatenate([train_game_ids[4:], test_game_ids]), train_game_ids[:4]
    else:
        raise Exception("Works only for fold 0")

    return test_game_ids, train_game_ids


def imread(fname):
    img = cv2.imread(fname)
    if img is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def imsave(fname, img, p=None):
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(fname, img, p)


def get_p_mat(y_pred, y_true, th=0.5):
    y_pred = (y_pred.detach().sigmoid().view(-1) > th).float()
    y_true = (y_true.view(-1) > 0.5).float()

    tp = (y_true * y_pred).sum()
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()

    return torch.stack([tp, tn, fp, fn])


def np_sigmoid(x):
    return 1/(1 + np.exp(-x))


def get_p_mat_np(y_pred, y_true, th=0.5, sigmoid=True):
    if sigmoid:
        y_pred = np_sigmoid(y_pred)
    y_pred = (y_pred.reshape(-1) > th).astype(np.float32)
    y_true = (y_true.reshape(-1) > 0.5).astype(np.float32)

    tp = (y_true * y_pred).sum()
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()

    return np.stack([tp, tn, fp, fn]).astype(np.int)


def get_f1(tp, tn, fp, fn, epsilon=1e-7):
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision*recall) / (precision + recall + epsilon)

    return f1


def get_acc(tp, tn, fp, fn):
    return tp / (tp + fn), tn / (tn + fp)


def video2imgs(video_path, scale=1):
    vc = cv2.VideoCapture(video_path)

    width, height = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    depth = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    if scale != 1:
        height, width = int(scale*height), int(scale*width)

    # v_segm = np.zeros((depth, height, width), dtype=np.uint8)
    v_imgs = np.zeros((depth, height, width, 3), dtype=np.uint8)
    # v_boxes = np.zeros((depth, height, width, 3), dtype=np.uint8)

    f = 0
    while True:
        it_worked, img = vc.read()
        if not it_worked:
            break

        if scale != 1:
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_LANCZOS4 if scale < 1 else cv2.INTER_CUBIC)

        v_imgs[f, ...] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        f += 1

    vc.release()

    return v_imgs


def get_grid(boxes, labels, grid_sz=(56, 56), img_sz=(224, 224), cell_sz=None):
    if cell_sz is None:
        cell_sz = [img_sz[0] // grid_sz[0], img_sz[1] // grid_sz[1]]

    grid = np.zeros((6, grid_sz[0], grid_sz[1]))

    if len(boxes) > 0:
        boxes_c_y = ((boxes[:, 0] + boxes[:, 2]) / 2 // cell_sz[0]).astype(np.int)
        boxes_c_x = ((boxes[:, 1] + boxes[:, 3]) / 2 // cell_sz[1]).astype(np.int)

        boxes_h = (boxes[:, 2] - boxes[:, 0]) / img_sz[0]
        boxes_w = (boxes[:, 3] - boxes[:, 1]) / img_sz[1]

        boxes_y = ((boxes[:, 0] + boxes[:, 2]) / 2) % cell_sz[0] / cell_sz[0]
        boxes_x = ((boxes[:, 3] + boxes[:, 1]) / 2) % cell_sz[1] / cell_sz[1]

        grid[0, boxes_c_y, boxes_c_x] = 1
        grid[1, boxes_c_y, boxes_c_x] = boxes_y
        grid[2, boxes_c_y, boxes_c_x] = boxes_x
        grid[3, boxes_c_y, boxes_c_x] = boxes_h
        grid[4, boxes_c_y, boxes_c_x] = boxes_w
        grid[5, boxes_c_y, boxes_c_x] = (labels == 1) | (labels == 2)

    return grid


def _int(x):
    return x.round().astype(np.int)


def get_bboxes_v1(p_grid, gf, f_p_bmask, f_p_bb, f_p_pos, pos_type,
                  img_sz, cell_sz):
    f_p_coords = p_grid[gf, 1:-1]

    c_y, c_x, h, w = f_p_coords[:, f_p_bmask]

    ys, xs = np.where(f_p_bmask)

    h, w = h*img_sz[0], w*img_sz[1]

    y, x = (ys + c_y)*cell_sz, (xs + c_x)*cell_sz

    top, left, bottom, right = (_int(y - h//2), _int(x - w//2),
                                _int(y + h//2), _int(x + w//2))

    b_pos = np.zeros((f_p_bmask.sum(),), dtype=np.float32)
    for i in range(len(b_pos)):
        tc, bc, lc, rc = (max(0, top[i]//cell_sz), bottom[i]//cell_sz,
                          max(0, left[i]//cell_sz), right[i]//cell_sz)
        if tc == bc or lc == rc:
            pos_type = 'point'

        if pos_type == 'point':
            b_pos[i] = f_p_pos[int((top[i] + bottom[i])/2/cell_sz),
                               int((left[i]+right[i])/2/cell_sz)]
        elif pos_type == 'max':
            b_pos[i] = f_p_pos[tc:bc, lc:rc].max()
        elif pos_type == 'mean':
            b_pos[i] = f_p_pos[tc:bc, lc:rc].mean()

    b_bb = f_p_bb[f_p_bmask]

    return b_bb, top, left, bottom, right, b_pos


def get_bboxes_v2(dets, helm_th2, f, pos_map, pos_type, cell_sz):
    boxes = dets[f].numpy()[:,:4].copy()
    scores = dets[f].numpy()[:,4].copy()

    indexes = np.where((scores > helm_th2))[0]
    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
    boxes = boxes[indexes].astype(np.int32).clip(min=0, max=511)

    top, left, bottom, right = [np.zeros((len(boxes),), dtype=np.int)
                                for _ in range(4)]
    for i, box in enumerate(boxes):
        top[i] = box[1] * 720 / 512
        left[i] = box[0] * 1280 / 512
        bottom[i] = box[3] * 720 / 512
        right[i] = box[2] * 1280 / 512

    b_pos = np.zeros((len(boxes),), dtype=np.float32)
    for i in range(len(b_pos)):
        tc, bc, lc, rc = (max(0, top[i]//4), bottom[i]//4,
                          max(0, left[i]//4), right[i]//4)
        sh = pos_map[tc:bc, lc:rc].shape
        if (sh[0] == 0) or (sh[1] == 0):
            pos_type = 'point'

        if pos_type == 'point':
            b_pos[i] = pos_map[int((top[i] + bottom[i])/2/cell_sz),
                               int((left[i]+right[i])/2/cell_sz)]
        elif pos_type == 'max':
            b_pos[i] = pos_map[tc:bc, lc:rc].max()
        elif pos_type == 'mean':
            b_pos[i] = pos_map[tc:bc, lc:rc].mean()

    b_bb = scores[indexes]

    return b_bb, top, left, bottom, right, b_pos


def get_final_pred_bboxes(
    all_pos_boxes, final_iou_threshold=0.3,
    score_field='bb_score', supress_range=5):
    pred_boxes_df = pd.DataFrame(
        all_pos_boxes, columns=[
            'frame', 'left', 'top', 'right', 'bottom',
            'bb_score', 'pos_score'])
    pred_boxes_df['processed'] = False
    pred_boxes_df['supressed'] = False

    while True:
        a_pred_boxes_df = pred_boxes_df.query("processed == False and supressed == False")

        if len(a_pred_boxes_df) == 0:
            break

        max_idx = a_pred_boxes_df[score_field].argmax()
        max_idx = a_pred_boxes_df.iloc[[max_idx]].index  # real index

        m_bbox = pred_boxes_df.loc[max_idx]

        frame = m_bbox['frame'].values[0]
        max_score = m_bbox[score_field].values[0]
        s_pred_boxes_df = pred_boxes_df.query("processed == False and supressed == False and abs(frame - @frame) <= @supress_range and frame != @frame")
        # s_pred_boxes_df = pred_boxes_df.query("processed == False and supressed == False and abs(frame - @frame) <= @supress_range and bb_score < @max_score")
        s_ious = torchvision.ops.box_iou(
            torch.from_numpy(m_bbox[['left', 'top', 'right', 'bottom']].values),
            torch.from_numpy(s_pred_boxes_df[['left', 'top', 'right', 'bottom']].values)
        )[0].numpy()

        s_idxs = s_pred_boxes_df[s_ious > final_iou_threshold].index

        pred_boxes_df.loc[s_idxs, 'supressed'] = True
        pred_boxes_df.loc[max_idx, 'processed'] = True

    pred_bboxes = pred_boxes_df.query("supressed == False")[['frame', 'left', 'top', 'right', 'bottom', 'bb_score', 'pos_score']].copy()
    pred_bboxes['frame'] += 1

    return pred_bboxes


def iou(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def get_final_pred_bboxes_v2(
    all_boxes, min_iou_threshold=0.5,
    score_field='bb_score', score_th=0.5, supress_range=5):
    all_boxes_df = pd.DataFrame(
        all_boxes, columns=[
            'frame', 'bb_score', 'top', 'left', 'bottom', 'right',
            'pos_score'])
    all_boxes_df['processed'] = False
    all_boxes_df['supressed'] = False

    while True:
        a_all_boxes_df = all_boxes_df.query(f"processed == False and supressed == False and {score_field}>@score_th")

        if len(a_all_boxes_df) == 0:
            break

        max_idx = a_all_boxes_df[score_field].argmax()
        max_idx = a_all_boxes_df.iloc[[max_idx]].index  # real index

        m_bbox = all_boxes_df.loc[max_idx]
        frame = m_bbox['frame'].values[0]
        max_score = m_bbox[score_field].values[0]

        #print('Current max:')
        #p_bbox(m_bbox)

        for r in [range(frame+1, frame+supress_range+1),
                  range(frame-1, frame-supress_range-1, -1)]:
            c_bbox = m_bbox[['left', 'top', 'right', 'bottom']].values[0]
            for f in r:
                #print(frame, f)
                s_all_boxes_df = all_boxes_df.query("frame == @f")
                t_bbox = s_all_boxes_df[['left', 'top', 'right', 'bottom']].values
                t_ious = []
                for b in t_bbox:
                    t_ious.append(iou(c_bbox, b))
                t_ious = np.array(t_ious)

                if len(t_ious) == 0:
                    #print(f'break len(t_ious) == 0')
                    break
                max_d = t_ious.argmax()
                #print(f'iou {t_ious[max_d]}')
                if t_ious[max_d] < min_iou_threshold:
                    #print(f'break min_iou {t_ious[max_d]}')
                    break
                max_d_idx = s_all_boxes_df.iloc[[max_d]].index

                if all_boxes_df.loc[max_d_idx, 'processed'].values[0]:
                    print('!!!')
                    #break
                #print('Supressed')
                #p_bbox(all_boxes_df.loc[max_d_idx])
                all_boxes_df.loc[max_d_idx, 'supressed'] = True
                # print(cy, cx)
                c_bbox = all_boxes_df.loc[max_d_idx][['left', 'top', 'right', 'bottom']].values[0]

        all_boxes_df.loc[max_idx, 'processed'] = True

    pred_bboxes = all_boxes_df.query(f"supressed == False and {score_field}>@score_th")[['frame', 'left', 'top', 'right', 'bottom', 'bb_score', 'pos_score']].copy()
    pred_bboxes['frame'] += 1

    return pred_bboxes

'''
def load_base_model(model_name, model_weiths, device):
    model_fn = getattr(models, 'get_'+model_name)
    try:
        model = model_fn(device)
        model.load_state_dict(torch.load(model_weiths))
    except Exception as e:
        tmp_model = nn.Sequential(OrderedDict([('module', model_fn(device))]))
        tmp_model.load_state_dict(torch.load(model_weiths))
        model = tmp_model.module

    return model.to(device)
'''

def load_any_base_model(model_weiths, device):
    tmp = torch.load(model_weiths, map_location=device)

    first_conv_k = [k for k in tmp.keys() if k.find('s1.pathway0_stem.conv.weight') > -1][0]
    num_channels = tmp[first_conv_k].shape[-3]
    num_heads = len([k for k in tmp.keys() if k.find('grid_heads.') > -1]) // 2

    # I use only two types of models
    if num_channels == 3:
        if num_heads == 0:
            model_fn = models.get_i3d3fpn_bbox_v0
        else:
            model_fn = models.get_mh_i3d3fpn_bbox_v0
    else:
        if num_heads == 0:
            model_fn = models.get_i3dfpn_bbox_v0
        else:
            model_fn = models.get_mh_i3dfpn_bbox_v0

    if num_heads == 0:
        model = model_fn(device)
    else:
        model = model_fn(device, head_number=num_heads)

    try:
        model.load_state_dict(tmp)
    except Exception as e:
        tmp_model = nn.Sequential(OrderedDict([('module', model_fn(device))]))
        tmp_model.load_state_dict(tmp)
        model = tmp_model.module

    return model.to(device)
