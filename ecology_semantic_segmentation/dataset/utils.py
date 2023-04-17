# fish_segmentation.py Dataset

import shutil
import os
import glob
import traceback

import cv2
import numpy as np

WHITE = ((0, 0, 237), (181, 25, 255))
BLACKGREY = ((0, 0, 0), (180, 255, 35))

BBOX_ANNOTATION_FILES = [
            [
                "/home/hans/data/Machine learning training set (copy)/12-23-2019/original image/f92B.jpg",
                "/home/hans/data/Machine learning training set (copy)/12-23-2019/whole body/f92B.png",
                WHITE
            ],
            [
                "/home/hans/data/Machine learning training set (copy)/photos 1.30.2019/original image/f9aB.jpg",
                "/home/hans/data/Machine learning training set (copy)/photos 1.30.2019/whole body/f9aB.png",
                BLACKGREY
            ],
        ]

DATA_DIR = "/home/hans/data/bbox_to_segmentation_gt/"
for fls in BBOX_ANNOTATION_FILES:
    img = fls[0]
    
    try:
        os.mkdir(os.path.join(DATA_DIR, "original image"))
    except Exception:
        pass
    
    if not os.path.exists(img):
        print ("Skipping %s" % '/'.join(img.split('/')[-3:]))
        continue

    os.rename(img, os.path.join(DATA_DIR, "original image", img.split('/')[-1]))
    
    ATLEAST_ONE = False
    for directory in glob.glob(os.path.join(os.path.dirname(os.path.dirname(img)), '*')):
        part = directory.split('/')[-1]

        try:
            mask = glob.glob(os.path.join(directory, '*' + img.split('/')[-1].split('.')[0] + '*'))[0]
            ATLEAST_ONE = True
        except Exception:
            traceback.print_exc()
            if not ATLEAST_ONE:
                os.remove(os.path.join(DATA_DIR, "original image", img.split('/')[-1]))

        try:
            os.mkdir(os.path.join(DATA_DIR, part))
        except Exception:
            pass

        shutil.copyfile(mask, os.path.join(DATA_DIR, part, mask.split('/')[-1]))

    #os.copyfile(img, os.path.join(DATA_DIR, "original image", img.split('/')[[-1]))

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
