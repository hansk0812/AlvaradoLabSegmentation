import numpy as np

import cv2
import albumentations as A
from albumentations.augmentations.transforms import FancyPCA, HueSaturationValue, CLAHE, ColorJitter, Emboss, RandomFog

from albumentations.augmentations.geometric.functional import rotate
from albumentations.augmentations.blur.transforms import GaussianBlur, Defocus, ZoomBlur

def augment_fn(image, masks):

    transforms = A.Compose([
                        A.OneOf([
                               Defocus(p=1),
                               GaussianBlur(p=1),
                               ZoomBlur(p=1),
                               RandomFog(p=0.4),
                           ], p=0.2),
                        A.OneOf([
                                ColorJitter(hue=0.4, brightness=0.4, contrast=0.4, saturation=0.4, p=0.3),
                                A.RandomBrightnessContrast(p=0.5),
                                A.RandomGamma(p=0.5),
                                Emboss(alpha=(0.3, 0.6), strength=(0.3, 0.7), p=0.3),
                            ], p=0.3),
#                        A.OneOf([
#                            A.RandomCrop(width=128, height=128, p=0.5),
#                            A.RandomCrop(width=64, height=64, p=0.5),
#                            ], p=0.5),
# Batch Size 1 for random sizes training
                       A.HorizontalFlip(p=0.5),
                       ], p=0.7)

    transform = transforms(image=image, masks=[masks[:,:,idx:idx+1] for idx in range(masks.shape[2])])
    image, masks = transform["image"], np.array(transform["masks"]).transpose((1,2,0,3))[...,0]
    
    image, masks = Arotate(image, masks, p=1)
    
    return image, masks

def Arotate(image, masks, degree=None, p=0.5):

    if degree is None:
        degree = np.random.randint(90)
        if np.random.rand() <= 0.2:
            degree = 0
    else:
        assert 0 < degree < 90, "Invalid degree ranges for rotate augmentation!"

    if np.random.rand() <= p: 
        image = rotate(image, degree)
        masks = rotate(masks, degree)

    return image, masks

if __name__ == "__main__":

    from .fish import fish_train_dataset

    for data in fish_train_dataset:
        img, segment, fname = data
        print (img.shape, segment.shape, img.min(), img.max(), segment.min(), segment.max(), fname)
        img, segment = (img.transpose((1,2,0))*255).astype(np.uint8), \
                       segment.transpose((1,2,0)).astype(np.uint8)*255
        print ("saving f.png")
        cv2.imwrite('f.png', img)
        cv2.imwrite('fm.png', segment[:,:,0])
        print (img.shape, segment.shape, img.min(), img.max())
        img, segment = augment_fn(img, segment)
        print ("saving g.png")
        cv2.imwrite('g.png', img)
        cv2.imwrite('gm.png', segment)