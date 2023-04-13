from torch.nn import functional as F
import numpy as np

import torch
from torch import nn

logsoftmax = lambda x: F.log_softmax(x, dim=1)

def cross_entropy_loss(gt, pred, weight=0.3, bce=False):

    if not bce:
        ce = F.cross_entropy(pred, gt) 
    else:
        ce = F.binary_cross_entropy(pred, gt) #, weight=torch.tensor([1-weight, weight]).cuda())
        #ce = - (1-weight) * (gt*logsoftmax(pred)) - weight * (1-gt)*logsoftmax(1-pred)
        #ce *= 1e-5
    return torch.sum(ce)

def focal_loss(gt, pred, gamma=1.5, factor=0.1):
    fl = - gt*torch.pow(pred, gamma)*logsoftmax(pred) - (1-gt)*torch.pow(1-pred, gamma)*logsoftmax(1-pred)
    #print (fl)
    return factor * torch.sum(fl)

def dice_loss(gt, pred, generalized=False):

    if not generalized:
        dl_n = 2 * torch.sum(gt * pred)
        dl_d = torch.sum(gt * gt) + torch.sum(pred * pred)
    
        return 1.0 - (dl_n + 1e-7) / (dl_d + 1e-7)
    else:
        # dc = (\sum p*gt + \eta) / (\sum p + \sum gt + \eta) + \
        #       (\sum (1-p)*(1-gt) + \eta) / ((\sum (1-p) + \sum (1-gt)) + \eta)
        # dice_loss = 1 - dc
        
        G1, P1 = gt, pred 
        G0, P0 = (1-gt), (1-pred)

        dice_coeff_preds_fg = torch.sum(G1 * P1) + 1e-7
        dice_coeff_preds_bg = torch.sum(G0 * P0) + 1e-7
        dice_coeff_normalize_fg = torch.sum(G1) + torch.sum(P1) + 1e-7
        dice_coeff_normalize_bg = torch.sum(G0) + torch.sum(P0) + 1e-7
        
        dc = (dice_coeff_preds_fg / dice_coeff_normalize_fg) + (dice_coeff_preds_bg / dice_coeff_normalize_bg)
        
        return 1.0 - dc

def twersky_loss(gt, pred, alpha=0.5, beta=0.3):

    tl_n = torch.sum(gt * pred) 
    tl_d = torch.sum(gt * pred) + alpha * torch.sum((1-pred)*gt) + beta * torch.sum(pred*(1-gt))

    return 1.0 - (tl_n + 1e-7) / (tl_d + 1e-7)

def focal_dice_coefficient(gt, pred, alpha=0.5, beta=0.3, gamma=1.8):

    fdsc_n = 2 * torch.sum(gt * torch.pow(1-pred, gamma) * pred) 
    fdsc_d = torch.sum(gt * torch.pow(1-pred, gamma) * pred) + \
             torch.sum((1-gt) * torch.pow(1-pred, alpha) * pred) + \
             torch.sum((1-pred) * torch.pow(pred, alpha) * gt)

    return 1.0 - (fdsc_n + 1e-7) / (fdsc_d + 1e-7)


def classification_dice_loss(gt, pred, factor=1e3):
    
    dice_l = dice_loss(gt, pred)
    generalized_dice_l = dice_loss(gt, pred, generalized=True)
    twersky_l = twersky_loss(gt, pred)
    focal_dice_l = focal_dice_coefficient(gt, pred) 
    m = factor * 0.33 
    return  dice_l*m, generalized_dice_l*m, twersky_l*m, focal_dice_l*m
