BACKGROUND_FLAG = False

from torch.nn import functional as F
import numpy as np

import torch
from torch import nn

from . import binary_cross_entropy

logsoftmax = lambda x: F.log_softmax(x, dim=1)

#binary_cross_entropy_list = lambda xL, yL: torch.sum([cross_entropy_loss(x, y, bce=True) for (x, y) in zip(xL, yL)])

def binary_cross_entropy_list(gt, pred):
    
    sum_arr = torch.zeros(6)
    for idx, (y, p) in enumerate(zip(gt, pred)):
        ce = cross_entropy_loss(y, p, bce=True)
        sum_arr[idx] = ce 

    return torch.sum(sum_arr)

cross_entropy_list = lambda xL, yL: torch.sum([cross_entropy_loss(x, y) for (x, y) in zip(xL, yL)])
focal_list = lambda xL, yL: torch.sum([focal_loss(x, y, bce=True) for (x, y) in zip(xL, yL)])
classification_dice_list = lambda xL, yL: torch.sum([classification_dice_loss(x, y, bce=True, background_weight=1) for (x, y) in zip(xL, yL)])

def cross_entropy_loss(gt, pred, weight=0.3, bce=False, background_weight = 0):

    if not bce:
        ce = F.cross_entropy(pred, gt) 
    else:   
        
        # more stable
        ce = binary_cross_entropy(pred, gt) 
        
        # one vs all loss functions for conceptual loss definition
        #ce += background_weight * binary_cross_entropy(1-pred, 1-gt)

        #eps = 1e-5
        #ce = F.binary_cross_entropy(pred, gt) # Tried + eps to remove nan 
        #ce = - (1-weight) * (gt*logsoftmax(pred)) - weight * (1-gt)*logsoftmax(1-pred)
        #ce *= 1e-5

    return torch.mean(ce)

def focal_loss(gt, pred, gamma=1.5, factor=0.1, background_weight=0):
    fl = - torch.pow((1-pred), gamma)*torch.log(pred+1e-7)
    
    if BACKGROUND_FLAG:
        fl += - background_weight * torch.pow(pred, gamma) * torch.log(1-pred+1e-7)

    return factor * torch.mean(fl)

def dice_loss(gt, pred, generalized=False, background_weight = 1):

    if not generalized:
        dl_n = 2 * torch.sum(gt * pred)
        dl_d = torch.sum(gt + pred * pred) # gt * gt = gt
        dice_fg = (dl_n + 1e-7) / (dl_d + 1e-7) 
     
        if BACKGROUND_FLAG:
            l_bg_n = 2 * torch.sum((1-gt)*(1-pred))
            l_bg_d = 2 * torch.sum((1-gt) + (1-pred)*(1-pred)) # 1-gt * 1-gt = 1-gt
            ice_bg = (dl_bg_n + 1e-7) / (dl_bg_d + 1e-7) 
            return - dice_fg - background_weight * dice_bg
        
        return - dice_fg

    else:
        # dc = (\sum p*gt + \eta) / (\sum p + \sum gt + \eta) + \
        #       (\sum (1-p)*(1-gt) + \eta) / ((\sum (1-p) + \sum (1-gt)) + \eta)
        # dice_loss = 1 - dc
        
        G1, P1 = gt, pred 
        G0, P0 = (1-gt), (1-pred)

        dice_coeff_preds_fg = torch.sum(G1 * P1) + 1e-7
        dice_coeff_normalize_fg = torch.sum(G1 + P1*P1) + 1e-7 # G1*G1 = G1
        dc = (dice_coeff_preds_fg / dice_coeff_normalize_fg) 
        
        if BACKGROUND_FLAG:
            dice_coeff_preds_bg = torch.sum(G0 * P0) + 1e-7
            dice_coeff_normalize_bg = torch.sum(G0 + P0*P0) + 1e-7 # G0*G0 = G0
            dc += background_weight * (dice_coeff_preds_bg / dice_coeff_normalize_bg)
        
        return - dc

def twersky_loss(gt, pred, alpha=0.5, beta=0.3, background_weight=0):

    tl_n = torch.sum(gt * pred) 
    tl_d = torch.sum(gt * pred) + alpha * torch.sum((1-pred)*gt) + beta * torch.sum(pred*(1-gt))
    td_fg = - (tl_n + 1e-7) / (tl_d + 1e-7)
    
    if BACKGROUND_FLAG:
        gt = 1-gt
        pred = 1-pred
        tl_bg_n = torch.sum(gt * pred) 
        tl_bg_d = torch.sum(gt * pred) + alpha * torch.sum((1-pred)*(gt)) + beta * torch.sum(pred*(1-gt))
        td_bg = - (tl_bg_n + 1e-7) / (tl_bg_d + 1e-7)
        return td_fg + background_weight * td_bg
    
    return td_fg

def focal_dice_coefficient(gt, pred, alpha=0.5, beta=0.3, gamma=1.8, background_weight=0):

    dl_n = 2 * torch.sum(gt * pred)
    dl_d = torch.sum(gt + pred * pred) # gt * gt = gt
    dice_coeff_fg = (dl_n + 1e-7) / (dl_d + 1e-7)
    fg_dice = - torch.pow(1-dice_coeff_fg, gamma) * torch.log(dice_coeff_fg + 1e-7)

    if BACKGROUND_FLAG:
        dl_bg_n = 2 * torch.sum((1-gt) * (1-pred))
        dl_bg_d = torch.sum((1-gt) + (1-pred) * (1-pred)) # 1-gt * 1-gt
        dice_coeff_bg = (dl_bg_n + 1e-7) / (dl_bg_d + 1e-7)
        bg_dice = - torch.pow(1-dice_coeff_bg, gamma) * torch.log(dice_coeff_bg + 1e-7)
        return fg_dice + background_weight * bg_dice

    return fg_dice

def classification_dice_loss(gt, pred, factor=1e3, background_weight=1):
    
    dice_l = dice_loss(gt, pred, background_weight=background_weight)
    generalized_dice_l = dice_loss(gt, pred, generalized=True, background_weight=background_weight)
    twersky_l = twersky_loss(gt, pred, background_weight=background_weight)
    focal_dice_l = focal_dice_coefficient(gt, pred, background_weight=background_weight) 
    m = factor * 0.33 
    return  dice_l*m, generalized_dice_l*m, twersky_l*m, focal_dice_l*m
