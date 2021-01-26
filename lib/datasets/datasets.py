from .patches_dataset import PatchesDataset
from .fragments_dataset_bbox import FragmentsDataset
from .wholeimage_dataset_bbox import (WholeImgDataset, WholeImgInferenceDataset,
                                      WholeImg512InferenceDataset,
                                      WholeRawImgDataset)


def get_patches_dataset(v_imgs, f_span, f_step, h_span, h_step,
                        w_span, w_step):
    return PatchesDataset(v_imgs, f_span, f_step, h_span,
                          h_step, w_span, w_step)


def get_fragments_dataset(fragments_path, fragments, cnt=8,
                          img_sz=224, transforms=None, return_grid=True):
    return FragmentsDataset(fragments_path, fragments, cnt,
                            img_sz, transforms, return_grid)


def get_wholeimage_dataset(data_path, labels, cnt=8, transforms=None,
                           return_grid=True, add_all=False):
    return WholeImgDataset(data_path, labels, cnt,
                            transforms, return_grid, add_all)


def get_wholeimage_inference_dataset(imgs, cnt=8):
    return WholeImgInferenceDataset(imgs, cnt)


def get_wholeimage512_inference_dataset(imgs):
    return WholeImg512InferenceDataset(imgs)


def get_wholerawimage_dataset(imgs, preprocess_fn, pad=True):
    return WholeRawImgDataset(imgs, preprocess_fn, pad)
