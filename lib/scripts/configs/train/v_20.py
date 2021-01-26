import os

base_model = 'i3d3fpn_bbox_v0'
model_weights = os.path.join(os.path.dirname(__file__), '../../../../checkpoints/v_2_f0_e3.pth')
img_sz = 224
cnt = 8
batch_size = 8
add_all = False
transforms = 'train_transforms_bbox_v0'
init_lr = 0.001
box_weight = 5.0
pos_weight = 1.0
loss_weights = [10.0, 2.0, 1.0]
class_mask = True
exp_frames = [-1,1]
epochs = 4
lr_update_epochs = [4,]
