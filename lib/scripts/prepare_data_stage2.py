import cv2

import sys
import os

import argparse
from functools import partial

from tqdm.auto import tqdm

import multiprocessing as mp

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from lib.utils import imsave, get_print_fn

from lib.paths import DATA_PATH, WHOLE_IMG_PATH


def process_video(video_name, videos_path, output_path, jpeg=True):
    v_output_path = video_name.split('.')[0]
    v_output_path = os.path.join(output_path, v_output_path)
    os.makedirs(v_output_path, exist_ok=True)

    video_path = os.path.join(videos_path, video_name)
    vc = cv2.VideoCapture(video_path)

    f = 0
    while True:
        it_worked, img = vc.read()

        if not it_worked:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if jpeg:
            imsave(os.path.join(v_output_path, f'{f}.jpg'), img,
                    p=[int(cv2.IMWRITE_JPEG_QUALITY), 90])
        else:
            imsave(os.path.join(v_output_path, f'{f}.png'), img)

        f += 1

    vc.release()


def main(output_path, p=4):
    exp_name = "data_prep_stage2"
    print_l = get_print_fn(exp_name)
    print_l(f"Start {exp_name}")

    videos_path = os.path.join(DATA_PATH, 'train')

    videos = os.listdir(videos_path)

    process_video_ = partial(
        process_video, videos_path=videos_path, output_path=output_path, jpeg=True)

    with mp.Pool(processes=p) as pool:
        for _ in tqdm(pool.imap_unordered(process_video_, videos), total=len(videos)):
            pass

    print_l(f"Finish {exp_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", dest="p",
                        help="number of processes", required=True, type=int)
    args = parser.parse_args()

    output_path = WHOLE_IMG_PATH
    os.makedirs(output_path, exist_ok=True)

    main(output_path, args.p)
