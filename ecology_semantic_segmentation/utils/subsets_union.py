import cv2
import numpy as np

import torch

from ..dataset.fish import ORGANS

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

def detect_inner_edges(pred, gt, img=None):
    # pred: Output from semantic segmentation NN with [0] - largest superset, [-1] - only set same as gt
    # pred is processed by return_union_sets_descending_order before this function is called
    # [BATCH, CLASSES, H, W]
    # gt.shape == pred.shape
    # img is None unless there is visualization based need
     
    if torch.cuda.is_available():
        pred = pred.cpu()
        gt = gt.cpu()

        if not img is None:
            img = img.cpu() * 255
            img = img.numpy().transpose((0,2,3,1)).astype(np.uint8)

    for b_idx in range(pred.shape[0]):
        for idx in range(pred.shape[1]-1):
            
            edges = detect_edges(img[b_idx], method="DoG")
            
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

            if not img is None:
                cv2.imshow("img", img[b_idx])

            edge_preds = set1 * (1-set1_gt)
            edge_pixels_inside_gt = edge_preds * set2_gt
            edge_pixels_outside_gt = edge_preds * (1-set2_gt)
            
            edge_preds_inner = (edge_pixels_inside_gt.numpy()*255).astype(np.uint8)
            edge_preds_outer = (edge_pixels_outside_gt.numpy()*255).astype(np.uint8)
            
            cv2.imshow("%s_pred_sub_gt_edges" % ORGANS[idx], ((
                edge_preds.numpy()*255).astype(np.uint8)))
            cv2.imshow("%s_edge_inside_gt_subset" % ORGANS[idx], edge_preds_inner)
            cv2.imshow("%s_edge_outside_gt_subset" % ORGANS[idx], edge_preds_outer)

            detect_edge_pred_overlap(edges, edge_preds_inner, "inner")
            detect_edge_pred_overlap(edges, edge_preds_outer, "outer")
            cv2.waitKey()

def detect_edges(img, method="sobel"):
    
    assert method in ["sobel", "canny", "DoG"]

    # sobel: [[-1,0,1],[-2,0,2],[-1,0,1]] and [[1,2,1],[0,0,0],[-1,-2,-1]]
    # canny: blur, sobel, nms, hysteresis (thresholding with 2 thresholds 
    # allowing yes/no inside range based on connectivity to 
    # strong edges : > threshold)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_filter_size = 3
    img_blur = cv2.GaussianBlur(img, (blur_filter_size,blur_filter_size), sigmaX=0, sigmaY=0)
    
    if method == "sobel":

        #sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=5, 
        #                    borderType=cv2.BORDER_ISOLATED, scale=2, delta=-1) # Sobel Edge Detection on the X axis
        #sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=5, 
        #                    borderType=cv2.BORDER_ISOLATED, scale=2, delta=-1) # Sobel Edge Detection on the Y axis
        
        edges = cv2.Sobel(src=img_blur, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5, 
                            borderType=cv2.BORDER_ISOLATED, scale=2, delta=-1) # Combined X and Y Sobel Edge Detection

    elif method == "DoG":
    
        blur1 = cv2.GaussianBlur(img, (5,5), 2.5)
        blur2 = cv2.GaussianBlur(img, (5,5), 2.15)

        edges = blur2 - blur1

        cv2.imshow('DoG', edges)
        cv2.waitKey()

    else:
        
        # perfect fish outlines but over-expression in the background
        #edges = cv2.Canny(image=img_blur, threshold1=200, threshold2=230, L2gradient=True, apertureSize=5) #aperture=7 works 
        
        # threshold1 = 10 gives more false positive edges around the backgrounds
        edges = cv2.Canny(image=img_blur, threshold1=30, threshold2=150, apertureSize=3) 
        
    return edges

def detect_edge_pred_overlap(edges, preds, edge_type="inner"):

    assert edge_type in ["inner", "outer"]
    edge_overlap = edges * preds
    cv2.imshow(edge_type, edge_overlap)

