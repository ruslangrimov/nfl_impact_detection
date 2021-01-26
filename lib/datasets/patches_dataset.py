import numpy as np

from torch.utils.data import Dataset


class PatchesDataset(Dataset):
    def __init__(self, v_imgs,
                 f_span, f_step,
                 h_span, h_step,
                 w_span, w_step):
        self.v_imgs = v_imgs

        depth, height, width = v_imgs.shape[:3]
        coordinats = []
        for f in range(0, depth, f_step):
            for h in range(0, height, h_step):
                for w in range(0, width, w_step):
                    coordinats.append((f, h, w))

        self.coordinats = coordinats
        self.f_span = f_span
        self.h_span = h_span
        self.w_span = w_span

    def __len__(self):
        return len(self.coordinats)

    def __getitem__(self, idx):
        f, h, w = self.coordinats[idx]
        f_span, h_span, w_span = self.f_span, self.h_span, self.w_span

        p_img = self.v_imgs[f:f+f_span, h:h+h_span, w:w+w_span]

        s = p_img.shape[:3]
        if s != (f_span, h_span, w_span):
            p_img = np.pad(p_img,
                           [[0, f_span-s[0]],
                            [0, h_span-s[1]],
                            [0, w_span-s[2]], [0, 0]],
                            mode='constant')

        p_img = p_img.transpose(3, 0, 1, 2).astype(np.float32) / 255.0
        coords = np.array([f, h, w], dtype=np.int)

        return p_img, coords
