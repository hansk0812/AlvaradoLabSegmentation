import cv2

import os
import glob

dir_paths = glob.glob(os.path.join("val_images/1910/", "*"))

for d in dir_paths:
    cv2.imshow(d, cv2.imread(d))
    cv2.waitKey()
