import os
import torch


def save_model(model, weight_path, ep, psnr):
    torch.save(model.state_dict(),
               os.path.join(weight_path, f"{ep}_{round(psnr, 4)}.pth"))
    return


def _get_psnr_from_filename(name):
    return float(name.split('_')[1].replace('.pth', ''))


def save_topk(model, ep, psnr, weight_path, topk_num):
    ckpts = os.listdir(weight_path)

    if len(ckpts) < topk_num:
        save_model(model, weight_path, ep, psnr)
    else:
        min_ckpt = min(ckpts, key=_get_psnr_from_filename)
        min_psnr = _get_psnr_from_filename(min_ckpt)
        if psnr > min_psnr:
            os.remove(os.path.join(weight_path, min_ckpt))
            save_model(model, weight_path, ep, psnr)
    return
