import cv2

import sys
import os

import argparse
import random
from functools import partial

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import multiprocessing as mp

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from lib.utils import imread, imsave, get_print_fn

from lib.paths import DATA_PATH, PREPARED_PATH


def get_video_and_segm(video_path, v_labels, scale=1):
    vc = cv2.VideoCapture(video_path)

    width, height = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    depth = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    if scale != 1:
        height, width = int(scale*height), int(scale*width)

    v_segm = np.zeros((depth, height, width), dtype=np.uint8)
    v_imgs = np.zeros((depth, height, width, 3), dtype=np.uint8)
    v_boxes = np.zeros((depth, height, width, 3), dtype=np.uint8)

    f = 0
    while True:
        it_worked, img = vc.read()

        if not it_worked:
            break

        if scale != 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

        v_imgs[f, ...] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = v_labels.query("frame == @f+1")
        for _, box in boxes.iterrows():
            if scale != 1:
                for a in ['top', 'left', 'height', 'width']:
                    setattr(box, a, int(scale*getattr(box, a)))

            if box.impact == 1:
                #lbl = 1
                if box.confidence > 1 and box.visibility > 0:
                    lbl = 1
                else:
                    lbl = 2
            else:
                lbl = 3

            v_segm[f, box.top:box.top+box.height, box.left:box.left+box.width] = lbl
            c_y, c_x = box.top + box.height // 2, box.left + box.width // 2
            v_boxes[f, c_y, c_x] = [lbl, box.height, box.width]

        f += 1

    vc.release()

    return v_imgs, v_segm, v_boxes


def save_patches(
        output_path,
        video_name, v_imgs, v_segm, v_boxes,
        f_step, f_span,
        h_step, h_span,
        w_step, w_span,
        pos=False, rand_start=True,
        jpeg=True):
    depth, height, width = v_segm.shape
    v_name = video_name.split('.')[0]

    if pos:
        cnt = 4
        fs = np.where((v_segm == 1).reshape(v_segm.shape[0], -1).any(-1))[0]
        fs = fs[fs > cnt]
        fs = (fs - cnt).tolist()  # If wh == 0???
    else:
        fs = range(random.randint(0, f_step // 4) if rand_start else 0, depth, f_step)

    for f in fs:
        # print(f)
        for h in range(random.randint(0, h_step // 4) if rand_start else 0, height, h_step):
            for w in range(random.randint(0, w_step // 4) if rand_start else 0, width, w_step):
                p_img = v_imgs[f:f+f_span, h:h+h_span, w:w+w_span]
                p_segm = v_segm[f:f+f_span, h:h+h_span, w:w+w_span]
                p_boxes =  v_boxes[f:f+f_span, h:h+h_span, w:w+w_span]

                if p_segm.sum() > 0:
                    h_o = h_span // 4
                    w_o = w_span // 4
                    #if (pos and (p_segm[f_span//2, h_o:-h_o, w_o:-w_o] == 1).sum() > 0) or\
                    # if (pos and (p_segm[:, h_o:-h_o, w_o:-w_o] == 1).sum() > 0) or\
                    if (pos and (p_segm[f_span//2, h_o:-h_o, w_o:-w_o] == 1).sum() > 0) or\
                        (not pos and (p_segm == 1).sum() == 0):
                        if not pos:
                            f_o = f_span // 4
                            p_img = p_img[-f_span:f_span, ...]
                            p_segm = p_segm[-f_span:f_span, ...]
                            p_boxes = p_boxes[-f_span:f_span, ...]

                        s = p_segm.shape
                        if s != (f_span, h_span, w_span):
                            p_segm = np.pad(p_segm,
                                            [[0, f_span-s[0]],
                                             [0, h_span-s[1]],
                                             [0, w_span-s[2]]],
                                            mode='constant')
                            p_img = np.pad(p_img,
                                           [[0, f_span-s[0]],
                                            [0, h_span-s[1]],
                                            [0, w_span-s[2]], [0, 0]],
                                            mode='constant')

                        f_name = f"{v_name}_{f}_{h}_{w}"
                        v_output_path = os.path.join(output_path, f_name)
                        os.makedirs(v_output_path, exist_ok=True)

                        for i in range(p_img.shape[0]):
                            if jpeg:
                                imsave(os.path.join(v_output_path, f'x_{i}.jpg'), p_img[i],
                                       p=[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                            else:
                                imsave(os.path.join(v_output_path, f'x_{i}.png'), p_img[i])
                            imsave(os.path.join(v_output_path, f'y_{i}.png'), p_segm[i])

                        wd, wy, wx = np.where(p_boxes[..., 0] > 0)
                        boxes_data = np.concatenate([np.stack([wd, wy, wx], axis=-1),
                                                     p_boxes[wd, wy, wx]], axis=-1)
                        np.save(os.path.join(v_output_path, f'boxes.npy'), boxes_data)


def process_video(video_name, videos_path, video_labels, output_path):
    h_span = 224
    w_span = 224

    video_path = os.path.join(videos_path, video_name)
    v_labels = video_labels.query("video == @video_name").copy()

    v_imgs, v_segm, v_boxes = get_video_and_segm(video_path, v_labels, scale=1)

    save_patches(
        os.path.join(output_path, 'pos'),
        video_name, v_imgs, v_segm, v_boxes,
        f_step=None,
        f_span=8,
        h_step=h_span//8,
        h_span=h_span,
        w_step=w_span//8,
        w_span=w_span, pos=True, rand_start=True, jpeg=True)

    save_patches(
        os.path.join(output_path, 'neg'),
        video_name, v_imgs, v_segm, v_boxes,
        f_step=6,
        f_span=16,
        h_step=h_span//2,
        h_span=h_span,
        w_step=w_span//2,
        w_span=w_span, pos=False, rand_start=True, jpeg=True)


def main(output_path, exp_fr=0, p=4):
    exp_name = "data_prep_stage1"
    print_l = get_print_fn(exp_name)
    print_l(f"Start {exp_name}")

    video_labels = pd.read_csv(os.path.join(DATA_PATH, 'train_labels.csv'))

    video_labels['impact'].sum()

    imp_video_labels = video_labels[video_labels['impact'] > 0].copy()
    print_l(f"Number of impacts {video_labels['impact'].sum()}")
    if exp_fr > 0:
        for _, row in tqdm(imp_video_labels.iterrows(), total=len(imp_video_labels)):
            frames = np.array(range(-exp_fr, exp_fr+1)) + row.frame
            video_labels.loc[(video_labels['video'] == row.video) &
                            (video_labels['frame'].isin(frames)) &
                            (video_labels['label'] == row.label), 'impact'] = 1

        print_l(f"Number of impacts after expansion {video_labels['impact'].sum()}")

    videos_path = os.path.join(DATA_PATH, 'train')

    videos = os.listdir(videos_path)

    process_video_ = partial(
        process_video, videos_path=videos_path, video_labels=video_labels,
        output_path=output_path)

    with mp.Pool(processes=p) as pool:
        for _ in tqdm(pool.imap_unordered(process_video_, videos), total=len(videos)):
            pass

    print_l(f"Finish {exp_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="output_path",
                        help="output path", required=True, type=str)
    parser.add_argument("-e", dest="exp_fr",
                        help="expand impacts", required=True, type=int)
    parser.add_argument("-p", dest="p",
                        help="number of processes", required=True, type=int)
    args = parser.parse_args()

    output_path = os.path.join(PREPARED_PATH, args.output_path)
    os.makedirs(output_path, exist_ok=True)

    main(output_path, args.exp_fr, args.p)
