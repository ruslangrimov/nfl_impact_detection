base_model = 'i3dfpn_bbox_v0'
data_folder = 'train_imgs_pos_8_224_224_v3_bboxes'
hard_mining = True
hard_segm_th = 0.1
hard_lbl_th = 0.1
img_sz = 224
cnt = 8
batch_size = 32
transforms = 'train_transforms_bbox_v0'
init_lr = 0.001
box_weight = 5.0
pos_weight = 1.0,
loss_weights = [10.0, 10.0, 1.0]
class_mask = True
epochs = 9
lr_update_epochs = [4,]