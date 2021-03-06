import os

base_model = 'i3dfpn_bbox_v0'
model_weights = os.path.join(os.path.dirname(__file__), '../../../../checkpoints/v_1_f0_e9.pth')
img_sz = 224
cnt = 8
batch_size = 8
add_all = False
transforms = 'train_transforms_bbox_v1'
init_lr = 0.0001
box_weight = 1.0
pos_weight = 1.0
loss_weights = [1.0, 1.0, 1.0]
class_mask = True
exp_frames = [-1,1]
epochs = 2
lr_update_epochs = [4,]
