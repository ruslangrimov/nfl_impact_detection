import os
import numpy as np

from torch.utils.data import Dataset

from lib.utils import imread, get_grid  # pylint: disable=import-error
from lib.datasets.augmentations import apply_transforms_bbox  # pylint: disable=import-error


class FragmentsDataset(Dataset):
    def __init__(
            self, fragments_path, fragments, cnt=8,
            img_sz=224, transforms=None, return_grid=True):
        self.fragments_path = fragments_path
        self.fragments = fragments
        self.cnt = cnt
        self.img_sz = img_sz
        self.transforms = transforms
        self.return_grid = return_grid

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx):
        f_path = os.path.join(os.path.join(self.fragments_path, 'neg'),
                              self.fragments[idx])
        if not os.path.isdir(f_path):
            f_path = os.path.join(os.path.join(self.fragments_path, 'pos'),
                                  self.fragments[idx])

        xs, ys = [], []

        for i in range(self.cnt):
            img = imread(os.path.join(f_path, f'x_{i}.jpg'))
            xs.append(img)
            img = imread(os.path.join(f_path, f'y_{i}.png'))
            ys.append(img[..., 0].astype(np.float32))

        boxes = np.load(os.path.join(f_path, 'boxes.npy'))

        boxes = (boxes[boxes[:, 0] == self.cnt//2][:, 1:]).astype(np.int)
        labels = boxes[:, 2]
        boxes = np.concatenate([boxes[:, :2], boxes[:, -2:]], axis=-1)

        pascal_boxes = np.stack([
                boxes[:, 1] - (boxes[:, 3] // 2).clip(1, None),
                boxes[:, 0] - (boxes[:, 2] // 2).clip(1, None),
                boxes[:, 1] + (boxes[:, 3] // 2).clip(1, None),
                boxes[:, 0] + (boxes[:, 2] // 2).clip(1, None),
            ], axis=-1).clip(0, self.img_sz)

        if self.transforms:
            xs, ys, pascal_boxes, labels = apply_transforms_bbox(
                xs, ys, pascal_boxes, labels, self.transforms)
            labels = np.array(labels)
            pascal_boxes = np.array(pascal_boxes)

        xs = np.stack(xs, axis=0)
        ys = np.stack(ys, axis=0)

        if len(pascal_boxes) > 0:
            boxes = pascal_boxes[:, [1, 0, 3, 2]] # xy to yx
        else:
            boxes = np.empty((0, 4), dtype=np.float32)

        assert len(boxes) == len(labels), "WTF?"

        xs = xs.transpose(3, 0, 1, 2).astype(np.float32) / 255.0

        segm = ((ys == 1) | (ys == 2)).astype(np.float32)[self.cnt//2]
        all_label = segm.any().astype(np.float32)[None]

        return (xs, all_label, get_grid(boxes, labels)) if self.return_grid\
            else (xs, all_label, boxes, labels)
