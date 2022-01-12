import os
import random
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from model.drln import make_model

from src.dataset import Dataset
from src.TrainFunction import train_step, valid_step
from src.utils import save_topk
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# python train.py --weights_fold weights3 --cuda cuda:1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_fold', type=str,
                        default='weights3', help='save the weights fold')
    parser.add_argument('--cuda', type=str,
                        default='cuda:1', help='cuda assigned')
    args = parser.parse_args()
    # parameters
    # =============
    epochs = 1500
    batch_size = 16
    scale = 3
    scale_type = '3x'  # '2x', '3x', '4x'
    patch = 32  # patch size in the low resolution image
    betas = (0.9, 0.999)
    init_lr = 1e-4
    weight_decay = 0
    eps = 1e-8
    step_size = 100
    gamma = 0.5
    save_topk_num = 5
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    # set up path
    # =============
    TrLR = os.path.join('datasets', 'Train', 'LR', scale_type)
    TrHR = os.path.join('datasets', 'Train', 'HR', scale_type)
    ValLR = os.path.join('datasets', 'Valid', 'LR', scale_type)
    ValHR = os.path.join('datasets', 'Valid', 'HR', scale_type)
    saveLoc = os.path.join('datasets', 'Valid', 'Result')
    if not os.path.exists(saveLoc):
        os.mkdir(saveLoc)
    weightLoc = os.path.join(args.weights_fold)
    if not os.path.exists(weightLoc):
        os.mkdir(weightLoc)

    # define model
    # =============
    model = make_model(scale).to(device)
    # model = DRLN(scale).to(device)

    # get train dataloader
    # =============
    TrDs = Dataset(LR_path=TrLR, HR_path=TrHR, scale=scale,
                   patch=patch)
    TrLoader = DataLoader(TrDs, batch_size=batch_size,
                          shuffle=True, num_workers=0, pin_memory=True)

    # loss function
    # =============
    loss_fn = nn.L1Loss()

    # optimizer
    # =============
    optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=betas,
                           weight_decay=weight_decay, eps=eps)

    # learning scheduler
    # =============
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=step_size,
                                    gamma=gamma)

    for epoch in range(1, epochs+1):
        print(f"Epoch: {epoch} / {epochs}")
        train_step(model, TrLoader, loss_fn, optimizer, device)
        scheduler.step()
        psnr = valid_step(model, loss_fn, device,
                          ValHR, ValLR, saveLoc)
        save_topk(model, epoch, psnr, weightLoc, save_topk_num)
