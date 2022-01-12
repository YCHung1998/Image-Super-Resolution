import os
from tqdm import tqdm

import torch
from torchvision.io import read_image
from torchvision.utils import save_image

from src.AccumulateAvg import AccAvg


def init_result():
    result = {
        'loss': AccAvg(),
        'psnr': AccAvg()
    }
    return result


def PSNR(SR, HR):
    return 10. * torch.log10((torch.mean((SR - HR) ** 2)**(-1)))


def train_step(model, Loader, loss_fn, optimizer, device):
    model.train()
    # print(device)
    result = init_result()
    bar = tqdm(Loader)
    for LR, HR in bar:
        LR, HR = LR.to(device), HR.to(device)

        optimizer.zero_grad()
        SR = model(LR)
        loss = loss_fn(SR, HR)
        loss.backward()
        optimizer.step()

        psnr = PSNR(SR, HR)
        result['loss'].update(loss.item(), LR.size(0))
        result['psnr'].update(psnr.item(), LR.size(0))

        # print the loss and psnr in tdqm bar
        record = {key: val.item() for key, val in result.items()}
        bar.set_postfix(record)


def valid_step(model, loss_fn, device, ValHRLoc, ValLRLoc, saveLoc):
    model.eval()  #.to('cpu')
    result = init_result()
    nameList = tqdm(os.listdir(ValHRLoc))
    for name in nameList:
        with torch.no_grad():

            LR = read_image(os.path.join(ValLRLoc, name)) / 255.
            HR = read_image(os.path.join(ValHRLoc, name)) / 255.

            LR = LR.unsqueeze(0).to(device)
            # LR = LR.unsqueeze(0)

            HR = HR.unsqueeze(0).to(device)
            # HR = HR.unsqueeze(0)
        
            SR = model(LR)
            loss = loss_fn(SR, HR)
            psnr = PSNR(SR, HR)
            print(psnr)

            # result['loss'].update(loss.item(), LR.size(0))
            result['psnr'].update(psnr.item(), LR.size(0))
            #save_image(SR, os.path.join(saveLoc, name))

            record = {key: val.item() for key, val in result.items()}
            nameList.set_postfix(record)
            
        del LR, HR, SR
    return result['psnr'].item()
