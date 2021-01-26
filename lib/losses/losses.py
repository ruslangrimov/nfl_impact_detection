import torch
import torch.nn as nn

from catalyst.contrib.nn.criterion import FocalLossBinary
from catalyst.contrib.nn.criterion import HuberLoss

class DetectionLoss(nn.Module):
    def __init__(self, box_weight=10.0, pos_weight=1.0, loss_weights=[1.0, 1.0, 1.0],
                 class_mask=True):
        super(DetectionLoss, self).__init__()
        self.bin_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(box_weight))
        self.bbox_criterion = HuberLoss()
        self.class_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        # self.class_criterion = FocalLossBinary(alpha=0.9, gamma=2)
        self.w = loss_weights
        self.class_mask = class_mask

    def forward(self, pred_grid, grid):
        bin_loss = self.bin_criterion(pred_grid[:, 0], grid[:, 0])
        b_mask = (grid[:, 0] > 0).unsqueeze(1)

        bbox_mask = b_mask.expand(-1, 4, -1, -1)
        bbox_loss = self.bbox_criterion(pred_grid[:, 1:5][bbox_mask][None].sigmoid(),
                                        grid[:, 1:5][bbox_mask][None])

        if self.class_mask:
            class_loss = self.class_criterion(pred_grid[:, 5:][b_mask][None],
                                              grid[:, 5:][b_mask][None])
        else:
            class_loss = self.class_criterion(pred_grid[:, -1], grid[:, -1])
        # print(bin_loss.item(), bbox_loss.item(), class_loss.item())
        loss = self.w[0] * bin_loss + self.w[1] * bbox_loss + self.w[2] * class_loss
        return loss
