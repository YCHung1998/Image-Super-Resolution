import os
import random
import torch

import torchvision.transforms.functional as f
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, scale, LR_path, HR_path, patch):
        '''
        scale = 3
        HR_path = os.path.join('datasets', 'Train', 'HR', '3x')
        '''
        self.scale = scale
        self.Path = [LR_path, HR_path]
        self.patch = patch
        self.name_list = os.listdir(self.Path[0])
        self.aug = Randaug()

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        # raw immage with LR HR
        img_raw = [read_image(os.path.join(path, name))
                   for path in self.Path]
        LR, HR = get_patch(img_raw[0], img_raw[1], self.patch, self.scale)
        LR, HR = self.aug.random_augment(LR, HR)
        # LR, HR = random_augment(LR, HR)
        LR = LR / 255.
        HR = HR / 255.
        return LR, HR

def pad(img, imgW, imgH, cropW, cropH, padW, padH):
    padding_ltrb = [
        (cropW - imgW) // 2 if padW else 0,
        (cropH - imgH) // 2 if padH else 0,
        (cropW - imgW + 1) // 2 if padW else 0,
        (cropH - imgH + 1) // 2 if padH else 0
    ]
    return f.pad(img, padding_ltrb, fill=0)

def get_patch(LR, HR, patch=64, scale=3):
    L_height, H_height = LR.size(1), HR.size(1)
    L_width, H_width = LR.size(2), HR.size(2)
    p_in = patch
    p_out = patch * scale

    # the flag flag decide pad or not
    padH = p_in > L_height
    padW = p_in > L_width

    # If LR is not large enough, pad.
    if padH or padW:
        LR = pad(LR, L_width, L_height, p_in, p_in, padW, padH)
        HR = pad(HR, H_width, H_height, p_out, p_out, padW, padH)

    # random crop
    L_height, H_height = LR.size(1), HR.size(1)
    L_width, H_width = LR.size(2), HR.size(2)
    lx = random.randrange(0, L_width - p_in + 1)
    ly = random.randrange(0, L_height - p_in + 1)
    hx = lx * scale
    hy = ly * scale
    cropLR = LR[:, ly:ly+p_in, lx:lx+p_in]
    cropHR = HR[:, hy:hy+p_out, hx:hx+p_out]

    return cropLR, cropHR

# aug_idx = np.arange(5)
# aug_list = [
#     lambda x: x,
#     lambda x: f.hflip(x),
#     lambda x: f.vflip(x),
#     lambda x: torch.rot90(x, k=1, dims=[1,2]),
#     lambda x: torch.rot90(x, k=2, dims=[1,2]),
#     lambda x: torch.rot90(x, k=3, dims=[1,2]),
# ]
# aug_dict = dict(zip(aug_idx, aug_list))

class Randaug(object):
    def __init__(self):
        self.aug_idx = [0, 1, 2, 3, 4, 5]
        self.aug_list = aug_list = [
            lambda x: x,
            lambda x: f.hflip(x),
            lambda x: f.vflip(x),
            lambda x: torch.rot90(x, k=1, dims=[1,2]),
            lambda x: torch.rot90(x, k=2, dims=[1,2]),
            lambda x: torch.rot90(x, k=3, dims=[1,2])
        ]
        self.aug_dict = dict(zip(self.aug_idx, self.aug_list))
    def random_augment(self, LR, HR):
        ID = random.randint(0, 5) 
        # if ID == 0:
        #     aug = lambda x: x
        # elif ID == 1:
        #     aug = lambda x: f.hflip(x)
        # elif ID == 2:
        #     aug = lambda x: f.vflip(x)
        # elif ID == 3:
        #     aug = lambda x: torch.rot90(x, k=1, dims=[1,2])
        # elif ID == 4:
        #     aug = lambda x: torch.rot90(x, k=2, dims=[1,2])
        # elif ID == 5:
        #     aug = lambda x: torch.rot90(x, k=3, dims=[1,2])
        LR = self.aug_dict[ID](LR)
        HR = self.aug_dict[ID](HR)
        return LR, HR
