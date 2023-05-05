# fish_segmentation.py Dataset

import shutil
import os
import glob
import traceback

import cv2
import numpy as np

# slightly slower than MSE - able to achieve results with MSE
#from skimage.metrics import structural_similarity as ssim

from .resources.composite_bboxes import BBOX_ANNOTATION_FILES

def return_bbox_corrected_annotations(imgpath, maskpath):
    img_np, mask_np = cv2.imread(imgpath), cv2.imread(maskpath)
    
    mask_binary = mask_np.copy()
    mask_binary[mask_binary > 225] = 0
    mask_binary[mask_binary > 0] = 1

    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    mask_binary = cv2.cvtColor(mask_binary, cv2.COLOR_BGR2GRAY)
    mask_np = cv2.cvtColor(mask_np, cv2.COLOR_BGR2GRAY)
    
    annotations[part] = mask_np 
    
    loc = np.zeros((img_np.shape[0]-mask_np.shape[0]+1, img_np.shape[1]-mask_np.shape[1]+1))
    for idx in range(0, img_np.shape[0]-mask_np.shape[0]+1):
        for jdx in range(0, img_np.shape[1]-mask_np.shape[1]+1):
            new_mask = img_np[idx:idx+mask_np.shape[0], jdx:jdx+mask_np.shape[1]] * mask_binary
            loc[idx][jdx] = np.sum((new_mask - mask_np)**2) #ssim(new_mask, mask_np, data_range=new_mask.max() - new_mask.min())

    min_idxs = np.unravel_index(loc.argmin(), loc.shape)
    
    mask_save = np.zeros_like(img_np)
    mask_save[min_idxs[0]: min_idxs[0]+mask_np.shape[0], min_idxs[1]:min_idxs[1]+mask_np.shape[1]] = mask_np
    #mask_save[mask_save>225] = 0 # white pixels as background?

    return mask_save

# Tried thresholding but found similarity based brute force bbox assignment to work better for "whole body" annotations
WHITE = ((0, 0, 237), (181, 25, 255))
BLACKGREY = ((0, 0, 0), (180, 255, 35))

ORIGINAL_DATA = "/home/hans/data/Machine learning training set/"
DATA_DIR = "/home/hans/data/bbox_to_segmentation_gt/"

original_images = [os.path.join(ORIGINAL_DATA, x) for x in  BBOX_ANNOTATION_FILES]

for idx, img in enumerate(original_images):
    
    if not os.path.exists(img):
        continue

    print ("Working on image - whole body ann pair %d/%d" %(idx+1, len(original_images)))
    print (img)
    try:
        os.mkdir(os.path.join(DATA_DIR, "original image"))
    except Exception:
        pass
    if not os.path.exists(img):
        print ("Skipping %s" % img)
        continue

    shutil.copyfile(img, os.path.join(DATA_DIR, "original image", img.split('/')[-1]))

    ATLEAST_ONE = False
    annotations = {}
    for directory in glob.glob(os.path.join(os.path.dirname(os.path.dirname(img)), '*')):
        part = directory.split('/')[-1]
        
        if part == "original image":
            continue
        try:
            os.mkdir(os.path.join(DATA_DIR, part))
        except Exception:
            pass
   
        try:
           
            mask = glob.glob(os.path.join(os.path.dirname(directory), part, '*' + img.split('/')[-1].split('.')[0] + '*'))[0]

            if part == "whole body":
                
                if os.path.exists(os.path.join(DATA_DIR, part, mask.split('/')[-1])):
                    continue

                shutil.copyfile(mask, os.path.join(DATA_DIR, part, mask.split('/')[-1]))
               
                mask_save = return_bbox_corrected_annotations(img, mask)

                cv2.imwrite(os.path.join(DATA_DIR, part, mask.split('/')[-1]), mask_save)
                os.remove(mask)

                ATLEAST_ONE = True

            else:
                
                #print ("Saving %s to %s" % (mask, os.path.join(DATA_DIR, part, mask.split('/')[-1])))
                shutil.copyfile(mask, os.path.join(DATA_DIR, part, mask.split('/')[-1]))
                os.remove(mask)

        except Exception:
            traceback.print_exc()
            pass

    os.remove(img)

# To ensure delete happens after checking the current state of newly created dataset
#exit()

for fl in original_images:
    img_name = fl.split('/')[-1].split('.')[0]
    mask_paths = glob.glob(os.path.join(os.path.dirname(os.path.dirname(fl)), "*", img_name+"*"))
    
    for f in mask_paths:
        os.remove(f)
