from torch.nn import functional as F
import numpy as np

import torch
from torch import nn

logsoftmax = lambda x: F.log_softmax(x, dim=2)

def cross_entropy_loss(gt, pred, weight=0.55):
    ce = F.binary_cross_entropy(pred, gt) 
    #F.cross_entropy(pred[:,0,:], torch.argmax(gt[:,0,:], dim=1)) #, weight=torch.tensor([1-weight, weight]).cuda())
    #ce = - (1-weight) * (gt*logsoftmax(pred)) - weight * (1-gt)*logsoftmax(1-pred)
    return torch.sum(ce)

def focal_loss(gt, pred, gamma=1.5, factor=0.1):
    fl = - gt*torch.pow(pred, gamma)*logsoftmax(pred) - (1-gt)*torch.pow(1-pred, gamma)*logsoftmax(1-pred)
    #print (fl)
    return factor * torch.sum(fl)

def dice_loss(gt, pred):
    dl_n = 2 * torch.sum(gt * pred)
    dl_d = torch.sum(gt * gt) + torch.sum(pred * pred)
    
    return 1.0 - (dl_n + 1e-7) / (dl_d + 1e-7)

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

    return factor * 0.33 * (
                   dice_loss(gt, pred) + \
                   twersky_loss(gt, pred) + \
                   focal_dice_coefficient(gt, pred))

