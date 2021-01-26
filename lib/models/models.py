import sys

class fakemodule(object):
    @staticmethod
    def ROIAlign(*args, **kwargs):
        print('ROIAlign')
        return None

# If we don't have detectron2 installed
if 'detectron2.layers' not in sys.modules:
    sys.modules['detectron2.layers'] = fakemodule

from . import i3dfpn_bbox

def get_i3dfpn_bbox_v0(device, add_4chan=True):
    return i3dfpn_bbox.get_model(device, model_type='i3d', add_4chan=add_4chan)

def get_i3dfpn_bbox_v1(device, add_4chan=True):
    return i3dfpn_bbox.get_model(device, model_type='i3d', add_4chan=add_4chan,
                                 out_channels=1)

def get_i3d3fpn_bbox_v0(device, add_4chan=True):
    return i3dfpn_bbox.get_model(device, model_type='i3d3', add_4chan=add_4chan)

def get_mh_i3dfpn_bbox_v0(device, head_number=4):
    model = get_i3dfpn_bbox_v0(device, add_4chan=True)
    return i3dfpn_bbox.MultiHeadModelWrapper(model, head_number)

def get_mh_i3d3fpn_bbox_v0(device, head_number=4):
    model = get_i3d3fpn_bbox_v0(device, add_4chan=True)
    return i3dfpn_bbox.MultiHeadModelWrapper(model, head_number)

def get_c2dfpn_bbox_v0(device, add_4chan=True):
    return i3dfpn_bbox.get_model(device, model_type='c2d', add_4chan=add_4chan)

def get_x3dsfpn_bbox_v0(device, add_4chan=True):
    return i3dfpn_bbox.get_model(device, model_type='x3ds', add_4chan=add_4chan)
