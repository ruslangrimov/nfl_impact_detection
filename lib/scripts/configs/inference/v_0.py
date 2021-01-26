from argparse import Namespace
import numpy as np

cfg = Namespace()

cfg.cnt = 8
cfg.scale = 1

cfg.grid_sz = (184, 320)
cfg.img_sz = (224, 224)
cfg.cell_sz = 4

cfg.non_blocking = False
cfg.batch_size = 4
cfg.num_workers = 2

cfg.bbox2 = False
cfg.bbox3 = True

cfg.helm_th = 0.40
cfg.use_helm_blobs = False
cfg._pos_th = 0.35
cfg.lbl_th = None

cfg.pos_type = 'point'
cfg.iou_threshold = 0.25

cfg.mult_pos_by_helm_scores = False
cfg.mult_helm_scores_by_pos_th = False

cfg.use_pos_max = False
cfg.final_use_pos_max = True

cfg.final_iou_threshold = 0.3
cfg.supress_range = 4

cfg.final_pred_bboxes_v2 = False
#cfg.final_iou_threshold = 0.3
#cfg.supress_range = 4
#cfg._pos_th = 0.45

# cfg.det2_weigths = DET_MODEL2_WEIGHTS
cfg.det3_models = ['resnet34',]
cfg.det3_weigths = ['helm_v1_f0_e11.pth',]
cfg.bbox3_ms = [0.75, 1.0, 1.25, 1.50]
ws = np.array([0.3, 1, 0.3, 0.15], dtype=np.float32)
#cfg.bbox3_ms = [1.0,]
#ws = np.array([1.0,], dtype=np.float32)
ws = ws / ws.sum()
cfg.bbox3_ws = ws.tolist()

cfg.base_weigths = ['model0.pth', 'model1.pth', 'model2.pth', 'model3.pth']
cfg.MEAN_MH = True

cfg.adv_pos_model = None
cfg.adv_pos_weights = None
cfg.adv_pos_a = 0.05

cfg.fix_th = 0.55
cfg.fix_dist = 1
