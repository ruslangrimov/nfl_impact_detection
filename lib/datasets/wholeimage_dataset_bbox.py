import os
import numpy as np

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from ..utils import imread, get_grid  # pylint: disable=import-error
from ..datasets.augmentations import apply_transforms_bbox  # pylint: disable=import-error

class WholeImgDataset(Dataset):
    def __init__(
            self, data_path, labels, cnt=8, transforms=None,
            return_grid=True, add_all=False
    ):
        self.scale = 1
        self.data_path = data_path
        self.labels = labels
        self.cnt = cnt
        self.transforms = transforms
        self.return_grid = return_grid

        videos = self.labels['video'].unique()

        self.items = []

        tmp = self.labels.groupby(['video', 'frame']).sum()

        self.items = []
        for video in videos:
            b_video = video.split('.')[0]
            total_fs = len(os.listdir(os.path.join(self.data_path, b_video)))
            for f in range(self.cnt // 2, total_fs - self.cnt // 2):
                tmp2 = tmp.loc[(video, f+1)]
                is_impact = tmp2['impact'] > 0 # and tmp2['confidence'] > 1 and tmp2['visibility'] > 0
                if add_all or is_impact or (np.random.randint(0, 5) == 3):
                    self.items.append((b_video, f, is_impact))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        b_video, cur_f, is_impact = self.items[idx]

        video_name = f'{b_video}.mp4'

        xs = []
        for f in range(cur_f-self.cnt//2, cur_f+self.cnt//2):
            xs.append(imread(os.path.join(self.data_path, b_video, f'{f}.jpg')))

        xs = np.stack(xs, axis=0)
        xs = np.pad(xs, [[0, 0], [0, 16], [0, 0], [0, 0]])

        pd_bboxes = self.labels.query("video == @video_name and frame == @cur_f + 1")\
            [['impact', 'left', 'width', 'top', 'height']]

        labels = (pd_bboxes['impact'] == 1).values.copy()

        pascal_boxes = np.stack([
            pd_bboxes['left'].values,
            pd_bboxes['top'].values,
            pd_bboxes['left'].values + pd_bboxes['width'].values,
            pd_bboxes['top'].values + pd_bboxes['height'].values,
        ], axis=-1)

        if self.transforms:
            ys = np.zeros(xs.shape[:-1], dtype=np.float32)
            xs, ys, pascal_boxes, labels = apply_transforms_bbox(
                xs, ys, pascal_boxes, labels, self.transforms)
            xs = np.array(xs)
            labels = np.array(labels)
            pascal_boxes = np.array(pascal_boxes)

        if len(pascal_boxes) > 0:
            boxes = pascal_boxes[:, [1, 0, 3, 2]] # xy to yx
        else:
            boxes = np.empty((0, 4), dtype=np.float32)

        assert len(boxes) == len(labels), "WTF?"

        xs = xs.transpose(3, 0, 1, 2).astype(np.float32) / 255.0

        all_label = labels.any().astype(np.float32)[None]

        return (xs, all_label, get_grid(
                boxes, labels,
                grid_sz=(184, 320),
                # img_sz=(736, 1280),
                img_sz=(224, 224),
                cell_sz=(4, 4),
            )) if self.return_grid\
            else (xs, all_label, boxes, labels)


class WholeImgInferenceDataset(Dataset):
    def __init__(self, imgs, cnt):
        super().__init__()
        self.imgs = imgs
        self.cnt = cnt

    def __len__(self) -> int:
        return len(self.imgs) - self.cnt

    def __getitem__(self, idx: int):
        image = self.imgs[idx:idx+self.cnt]
        h_p = (32 - image.shape[1] % 32) % 32
        #print('h_p', h_p)
        image = np.pad(image, [[0, 0], [0, h_p], [0, 0], [0, 0]])
        image = image.astype(np.float32) / 255.0
        image = image.transpose(3, 0, 1, 2)
        return image


class WholeRawImgDataset(Dataset):
    def __init__(self, imgs, preprocess_fn, pad=True):
        super().__init__()
        self.imgs = imgs
        self.preprocess_fn = preprocess_fn
        self.pad = pad

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        image = self.imgs[idx]
        if self.pad:
            h_p = (32 - image.shape[0] % 32) % 32
            #print('h_p', h_p)
            image = np.pad(image, [[0, h_p], [0, 0], [0, 0]])
        if self.preprocess_fn is not None:
            image = self.preprocess_fn(image)
        return image.transpose(2, 0, 1).astype(np.float32)


class WholeImg512InferenceDataset(Dataset):
    def __init__(self, imgs):
        super().__init__()
        self.imgs = imgs
        self.transforms = A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        image = self.imgs[idx].astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        return image
