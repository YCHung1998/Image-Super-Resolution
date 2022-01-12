import os
import torch
import torchvision.transforms.functional as f
from torchvision.io import read_image
from torchvision.utils import save_image

from model.drln import DRLN
import argparse

'''
python inference.py --weights_fold weights1 \
--weights 394_26.8162.pth --save_fold answer1_394 \
--transforms False
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_fold', type=str,
                        default='weights', help='weights fold')
    parser.add_argument('--weights', type=str,
                        default='895_31.3226.pth', help='weights')
    parser.add_argument('--save_fold', type=str,
                        default='answer', help='answer fold')
    parser.add_argument('--cuda', type=str,
                        default='cuda:1', help='cuda assigned')
    parser.add_argument('--transforms', type=bool,
                        default=True, help='transforms or not')
    args = parser.parse_args()

    # set up path
    testLoc = os.path.join('datasets', 'testing_lr_images')
    resultLoc = os.path.join(args.save_fold)
    if not os.path.exists(resultLoc):
        os.mkdir(resultLoc)
    ckpt = os.path.join(args.weights_fold, args.weights)
    testList = os.listdir(testLoc)

    # read model
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    model = DRLN(scale=3)
    model.load_state_dict(torch.load(ckpt))
    model.eval().to(device)

    # transforms
    if args.transforms:
        Tform = [(lambda x: x, lambda x: x),
                 (lambda x: f.hflip(x), lambda x: f.hflip(x)),
                 (lambda x: f.vflip(x), lambda x: f.vflip(x)),
                 (lambda x: torch.rot90(x, k=1, dims=[2, 3]),
                  lambda x: torch.rot90(x, k=3, dims=[2, 3])),
                 (lambda x: torch.rot90(x, k=2, dims=[2, 3]),
                  lambda x: torch.rot90(x, k=2, dims=[2, 3])),
                 (lambda x: torch.rot90(x, k=3, dims=[2, 3]),
                  lambda x: torch.rot90(x, k=1, dims=[2, 3])),
                 (lambda x: torch.rot90(f.hflip(x), k=1, dims=[2, 3]),
                  lambda x: torch.rot90(f.hflip(x), k=1, dims=[2, 3])),
                 (lambda x: torch.rot90(f.vflip(x), k=1, dims=[2, 3]),
                  lambda x: torch.rot90(f.vflip(x), k=1, dims=[2, 3])),
                 (lambda x: torch.rot90(f.hflip(x), k=2, dims=[2, 3]),
                  lambda x: torch.rot90(f.hflip(x), k=2, dims=[2, 3])),
                 (lambda x: torch.rot90(f.vflip(x), k=2, dims=[2, 3]),
                  lambda x: torch.rot90(f.vflip(x), k=2, dims=[2, 3])),
                 (lambda x: torch.rot90(f.hflip(x), k=3, dims=[2, 3]),
                  lambda x: torch.rot90(f.hflip(x), k=3, dims=[2, 3])),
                 (lambda x: torch.rot90(f.vflip(x), k=3, dims=[2, 3]),
                  lambda x: torch.rot90(f.vflip(x), k=3, dims=[2, 3]))]
    else:
        Tform = [
            (lambda x: x, lambda x: x),
        ]

    for imgname in testList:
        imgID = imgname.split('.')[0]

        img = read_image(os.path.join(testLoc, imgname)) / 255.
        _, H, W = img.shape
        img = img.unsqueeze(0).to(device)
        SR = torch.zeros(1, 3, 3*H, 3*W).to(device)

        for forward, backward in Tform:
            img = forward(img)
            with torch.no_grad():
                sr = model(img)
                sr = backward(sr)
                img = backward(img)
                SR += sr

        SR = SR / len(Tform)
        save_image(SR, os.path.join(resultLoc, imgID+'_pred.png'))
