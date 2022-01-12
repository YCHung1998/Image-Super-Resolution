import os
import shutil

from tqdm import tqdm
from PIL import Image


def renew_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        os.makedirs(path)
    return


def produce_pairs(HR, scale):
    '''
    pre-process the hr image size to be divisible to the scale.
    cut those columns and rows that mod scale is not equal 0.
    LR create by HR in bicubic method.
    '''
    W, H = HR.width, HR.height
    sW, sH = W - (W % scale), H - (H % scale)
    HR = HR.crop((0, 0, sW, sH))

    tW, tH = int(sW // scale), int(sH // scale)
    LR = HR.resize((tW, tH), Image.BICUBIC)
    return LR, HR

if __name__ == '__main__':
    Type = 'valid'

    RawImgLoc = os.path.join('datasets', 'validation_hr_images')

    TrainHR = os.path.join('datasets', 'Valid', 'HR')
    TrainLR = os.path.join('datasets', 'Valid', 'LR')
    renew_dir(TrainHR)
    renew_dir(TrainLR)
    scaleList = ['2x', '3x', '4x']
    for scale in scaleList:
        os.makedirs(os.path.join(TrainHR, scale))
        os.makedirs(os.path.join(TrainLR, scale))

    nameList = os.listdir(RawImgLoc)
    for name in tqdm(nameList):
        raw = Image.open(os.path.join(RawImgLoc, name))
        for scale, scaleFolder in enumerate(scaleList, start=2):
            LR, HR = produce_pairs(raw, scale)
            LR.save(os.path.join(TrainLR, scaleFolder, name))
            HR.save(os.path.join(TrainHR, scaleFolder, name))
