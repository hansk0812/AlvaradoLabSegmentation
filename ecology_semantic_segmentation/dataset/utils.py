# fish_segmentation.py Dataset

import shutil
import os
import glob
import traceback

import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim

WHITE = ((0, 0, 237), (181, 25, 255))
BLACKGREY = ((0, 0, 0), (180, 255, 35))

BBOX_ANNOTATION_FILES = [
                "12-23-2019/original image/f92B.jpg",
                "photos 1.30.2019/original image/f9aB.jpg",
                "photos 1.30.2019/original image/f97B.jpg",
                "12-23-2019/original image/f93B.jpg",
                "photos 1.30.2019/original image/f96B.jpg",
                "photos 1.30.2019/original image/f99B.jpg",
                "photos 1.30.2019/original image/f25B.jpg", 
                "photos 1.30.2019/original image/f6B.png", 
                "photos 1.30.2019/original image/f15aB.jpg", 
                "photos 1.30.2019/original image/f7B.png", 
                "photos 1.30.2019/original image/f99B.jpg", 
                "photos 1.30.2019/original image/f199B.jpg", 
                #"photos 1.30.2019/original image/f45B.png",
                #"photos 1.30.2019/original image/f34B.jpg",
                #"12-23-2019/original image/f58B.jpg",
                #"photos 1.30.2019/original image/f52B.jpg",
                        ]

ORIGINAL_DATA = "/home/hans/data/Machine learning training set (copy)/"
DATA_DIR = "/home/hans/data/bbox_to_segmentation_gt/"

for img in [os.path.join(ORIGINAL_DATA, x) for x in  BBOX_ANNOTATION_FILES]:
    
    try:
        os.mkdir(os.path.join(DATA_DIR, "original image"))
    except Exception:
        pass
    if not os.path.exists(img):
        print ("Skipping %s" % img)
        continue

    shutil.copyfile(img, os.path.join(DATA_DIR, "original image", img.split('/')[-1]))

    ATLEAST_ONE = False
    for directory in glob.glob(os.path.join(os.path.dirname(os.path.dirname(img)), '*')):
        part = directory.split('/')[-1]
        
        if part == "original image":
            continue
        try:
            os.mkdir(os.path.join(DATA_DIR, part))
        except Exception:
            pass
   
        try:
            mask = glob.glob(os.path.join(directory, '*' + img.split('/')[-1].split('.')[0] + '*'))[0]
            ATLEAST_ONE = True
            shutil.copyfile(mask, os.path.join(DATA_DIR, part, mask.split('/')[-1]))
            
            if part == "whole body":
                
                img_np, mask_np = cv2.imread(img, -1), cv2.imread(mask, -1)
                mask_binary = mask_np.copy()
                mask_binary[mask_binary > 225] = 0
                mask_binary[mask_binary > 0] = 1

                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                mask_binary = cv2.cvtColor(mask_binary, cv2.COLOR_BGR2GRAY)
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_BGR2GRAY) * mask_binary
                
                loc = np.zeros((img_np.shape[0]-mask_np.shape[0]+1, img_np.shape[1]-mask_np.shape[1]+1))
                for idx in range(0, img_np.shape[0]-mask_np.shape[0]+1):
                    for jdx in range(0, img_np.shape[1]-mask_np.shape[1]+1):
                        new_mask = img_np[idx:idx+mask_np.shape[0], jdx:jdx+mask_np.shape[1]] * mask_binary
                        loc[idx][jdx] = np.sum((new_mask - mask_np)**2) #ssim(new_mask, mask_np, data_range=new_mask.max() - new_mask.min())

                min_idxs = np.unravel_index(loc.argmin(), loc.shape)
                
                cv2.imshow('img', img_np)
                img_np[min_idxs[0]: min_idxs[0]+mask_np.shape[0], min_idxs[1]:min_idxs[1]+mask_np.shape[1]] = mask_np
                cv2.imshow('mask', img_np.astype(np.uint8))
                cv2.waitKey()

        except Exception:
            traceback.print_exc()
            pass

    #if not ATLEAST_ONE:
    #    os.remove(os.path.join(DATA_DIR, "original image", img.split('/')[-1]))

def create_mask(img_shape, img_file, mask_file, color=WHITE):
    # Create bounding box based on white background

    img = cv2.imread(img_file)
    
    if img is None:
        return 

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mask = cv2.imread(mask_file)
    mask = cv2.inRange(img, color[0], color[1])
    #mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    
    mask = 255 - mask
    where = np.array(np.where(mask))
    x1, y1 = np.amin(where, axis=1)
    x2, y2 = np.amax(where, axis=1)
    
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.rectangle(img, (y1, x1), (y2, x2), (255,0,0), 2)
     
    #mask = cv2.resize(mask, (y2-y1+1, x2-x1+1))
    mask_ret = np.zeros_like(img[:,:,:1])
    #mask_ret[x1:x2+1, y1:y2+1, 0] = mask
    
    cv2.imshow('f', img)
    cv2.imshow('g', mask)
    cv2.waitKey()

    cv2.imwrite("sample.png", mask)
    cv2.imwrite("sample_img.png", img)

for data in BBOX_ANNOTATION_FILES:
    imgpath, annpath, color = data[0], data[1], data[2]
    create_mask((256,256,3), imgpath, annpath, color)
