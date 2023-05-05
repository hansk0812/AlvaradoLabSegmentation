import cv2
import time
import numpy as np

from .manual_bbox_identification import return_bbox_corrected_annotations

# Found small artifacts inside segmentation file of "whole body"!
# 12-23-2019/original image/f99C.jpg

def remove_islands_in_segment_gt(mask):
    
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 1:
        large_enough_contours = []
        for idx, contour in enumerate(contours):
            area = shoelace_algorithm(contour)
            
            # Remove random spots on mask based on contour area
            if area > 1000:
                large_enough_contours.append(contour)
        
        mask = np.zeros_like(mask)
        mask = cv2.drawContours(mask, large_enough_contours, -1, 1, -1)

    return mask

def shoelace_algorithm(contour):
    
    N = len(contour)
    area = 0
    for idx, jdx in zip(range(N-1), range(1, N)):
        p1, p2 = contour[idx][0], contour[jdx][0]
        area += abs(p1[0]*p2[1] - p1[1]*p2[0])
    
    return area * 0.5

def get_bounding_box_from_mask(mask):
    
    mask = remove_islands_in_segment_gt(mask)

    maskx = np.any(mask, axis=0)
    masky = np.any(mask, axis=1)
    
    x1 = np.argmax(maskx)
    y1 = np.argmax(masky)
    
    x2 = len(maskx) - np.argmax(maskx[::-1])
    y2 = len(masky) - np.argmax(masky[::-1])
     
    return (x1, y1), (x2, y2)
    # return (xmin, ymin), (xmax, ymax)

if __name__ == "__main__":

    from .fish import fish_train_dataset, fish_val_dataset, fish_test_dataset
    
    # Display all files that need to be moved to new dataset folder
    for img, ann, fname in fish_test_dataset:

        ann[ann>0] = 1
        ann = ann[0]
        
        (x1, y1), (x2, y2) = get_bounding_box_from_mask(ann)
        
        threshold=5
        if abs((x2 - x1) - ann.shape[0]) +  abs((y2-y1) - ann.shape[1]) < 2*threshold:
            print (fname)

