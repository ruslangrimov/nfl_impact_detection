import albumentations as A


def get_train_transforms_bbox_v0(frames=8):
    # Hard color
    # Light scale, shift
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.ColorJitter(p=1),
                A.HueSaturationValue(40, 60, 40, p=1),
                A.RGBShift(60, 60, 60, p=1),
                A.RandomGamma((20, 200), p=1)
            ], p=0.8),
            A.OneOf([
                A.GaussianBlur(p=1),
                A.GaussNoise((10.0, 50.0), p=1),
                A.MedianBlur(blur_limit=5, p=1),
                A.MotionBlur(p=1),
            ], p=0.6),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0,
                            border_mode=0, p=0.4),
            # A.GridDistortion(p=0.2, border_mode=0),
        ],
        additional_targets=dict(
            **{f"image{i}": 'image' for i in range(1, frames)},
            **{f"mask{i}": 'mask' for i in range(1, frames)}
        ),
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        ),
        p=1.0)


def get_train_transforms_bbox_v1(frames=8):
    # Hard color
    # Light scale, shift
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.ColorJitter(p=1),
                A.HueSaturationValue(40, 60, 40, p=1),
                A.RGBShift(60, 60, 60, p=1),
                A.RandomGamma((20, 200), p=1)
            ], p=0.8),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0,
                            border_mode=0, p=0.4),
            # A.GridDistortion(p=0.2, border_mode=0),
        ],
        additional_targets=dict(
            **{f"image{i}": 'image' for i in range(1, frames)},
            **{f"mask{i}": 'mask' for i in range(1, frames)}
        ),
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        ),
        p=1.0)


def _ik(i):
    return f"image{i if i > 0 else ''}"


def _im(i):
    return f"mask{i if i > 0 else ''}"


def apply_transforms_bbox(imgs, masks, bboxes, labels, transforms):
    inp = {_ik(i): x for i, x in enumerate(imgs)}
    inp.update({_im(i): x for i, x in enumerate(masks)})
    inp['bboxes'] = bboxes
    inp['labels'] = labels

    out = transforms(**inp)

    imgs = [out[_ik(i)] for i in range(len(imgs))]
    if masks is not None:
        masks = [out[_im(i)] for i in range(len(masks))]

    return imgs, masks, out['bboxes'], out['labels']
