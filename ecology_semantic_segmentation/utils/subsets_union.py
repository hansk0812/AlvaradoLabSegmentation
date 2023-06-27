import cv2
import numpy as np

import torch

def return_union_sets_descending_order(ann, exclude_indices=[0], reverse=False):
    # exclude_indices: Eliminate composite segmentation unions to prevent learning the same segment
    # Preferred order: easiest to segment organ as ann[-1] --> hardest to segment as ann[sorted(idx) \ exclude_indices]
    # GT label ordering dependent: env variable: 
    #ORGANS needs sequence relevant ordering based on hardest-to-segment organs
    # Based on how the regularization made me decide to do this, this code isn't a dataset based xy pair trick
    # reverse: supersets to organs
     
    # torch polygon artefacts based on jagged edges
    if not reverse:
        for idx in range(ann.shape[1]-1):
            if idx in exclude_indices:
                continue
            ann[:,idx] = torch.sum(ann[:,idx:], axis=1)
        ann[ann>1] = 1
    else:
        for idx in range(ann.shape[1]-2, -1, -1):
            if idx in exclude_indices:
                continue
            ann[:,idx] = torch.abs(ann[:,idx]-ann[:,idx+1])
            
            # including edge activations for membership study - detect_inner_edges
            #ann[:,idx] = (ann[:,idx] > 200/255.).int()

    return ann

def detect_inner_edges(pred, gt):
    # pred: Output from semantic segmentation NN with [0] - largest superset, [-1] - only set same as gt
    # pred is processed by return_union_sets_descending_order before this function is called
    # [BATCH, CLASSES, H, W]
     
    if torch.cuda.is_available():
        pred = pred.cpu()
        gt = gt.cpu()

    for b_idx in range(pred.shape[0]):
        for idx in range(pred.shape[1]-1):
            set1, set2 = pred[b_idx,idx], pred[b_idx,idx+1]
            set1_gt, set2_gt = gt[b_idx,idx], gt[b_idx,idx+1]
            
            cv2.imshow("set1", ((
                set1.numpy()*255).astype(np.uint8)))
            cv2.imshow("set2", ((
                set2.numpy()*255).astype(np.uint8)))
            cv2.imshow("set1gt", ((
                set1_gt.numpy()*255).astype(np.uint8)))
            cv2.imshow("set2gt", ((
                set2_gt.numpy()*255).astype(np.uint8)))


            edge_preds = set1 * (1-set1_gt)
            edge_pixels_inside_gt = edge_preds * set2_gt
            edge_pixels_outside_gt = edge_preds * (1-set2_gt)
            
            cv2.imshow("edge_preds", ((
                edge_preds.numpy()*255).astype(np.uint8)))
            cv2.imshow("edge_inside_gt", ((
                edge_pixels_inside_gt.numpy()*255).astype(np.uint8)))
            cv2.imshow("edge_outside_gt", ((
                edge_pixels_outside_gt.numpy()*255).astype(np.uint8)))
            cv2.waitKey()

