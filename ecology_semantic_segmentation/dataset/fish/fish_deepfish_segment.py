import json 
import os
import glob
import traceback

import rawpy
import cv2
import numpy as np

from torch.utils.data import Dataset

from . import composite_labels

from ..augment import augment_fn
#from ..bbox_masks_problem import remove_islands_in_segment_gt

from ..bbox_masks_problem import shoelace_algorithm

#from .fish_segmentation import imread

class DeepFishDataset(Dataset):

    def __init__(self, segmentation_data, img_shape, min_segment_positivity_ratio, sample_dataset = True, organs=["whole_body"], augment_flag=False): 
        
        if sample_dataset:
            segmentation_data = {key: segmentation_data[key] for key in list(segmentation_data)[:60]}
        
        # Ensure all files contribute to data wrt organs
        if organs is None:
            test_organs = composite_labels
        else:
            test_organs = organs
            
        self.segmentation_data = segmentation_data
        self.segmentation_keys = list(segmentation_data.keys())
        self.img_shape = img_shape 
        self.min_segment_positivity_ratio = min_segment_positivity_ratio
        self.organs = organs
        
        self.set_augment_flag(augment_flag)

    def __len__(self):
        return len(self.segmentation_keys)
    
    def set_augment_flag(self, flag):
        self.augment_flag = flag

    def __getitem__(self, idx):
        
        data_key = self.segmentation_keys[idx]
        
        image_path, segment_path = self.segmentation_data[data_key]["image"], self.segmentation_data[data_key]["segment"]
        image = cv2.imread(image_path)
        image_shape = image.shape
        image = cv2.resize(image, (self.img_shape, self.img_shape))

        segment = np.zeros(image_shape, dtype=np.uint8)
        with open(segment_path, 'r') as f:
            data = json.load(f)
            
            # labels idx=0 contains 4/5 sided ROI
            #data["labels"] = data["labels"][1:] if len(data["labels"]) > 1 else data["labels"]
            for x in data["labels"]:
                pts = [np.expand_dims(np.array([(p["x"], p["y"]) for p in x["regions"][idx]], dtype=np.int32), axis=-2) \
                            for idx in range(len(x["regions"]))]
                pts = [x for x in pts if len(x) > 5]
                
                segment = cv2.fillPoly(segment, pts=pts, color=[255,255,255])
                #pts = [(p["x"], p["y"]) for p in x["polygons"]]
                #print (x["label_type"], x["anno_data"], x["regions"], len(pts))
            
            # image_filename, completed_tasks, labels
            segment = cv2.resize(segment, (self.img_shape, self.img_shape))[:,:,:1]
            image = cv2.resize(image, (self.img_shape, self.img_shape))
        
        #segment = remove_islands_in_segment_gt(segment)
        
        # Decide using this image: Machine learning training set (copy)/photos 1.30.2019/original image/f132C.png
        #SEGMENT_THRESHOLD = 225
        #segment[segment > SEGMENT_THRESHOLD] = 0
        #segment[segment != 0] = 255
        
        if self.augment_flag:
            image, segment = augment_fn(image, segment)
        
        return image.transpose((2,0,1)).astype(np.float32) / 255.0, segment.transpose((2,0,1)).astype(np.float32) / 255.0, image_path

def get_deepfish_segclsloc_data(dtype, path, folder_path, img_shape, min_segment_positivity_ratio, 
                                sample_dataset=True, organs=["whole_body"], bbox_dir=None ,augment_flag = True):
    
    #TODO: 9 missing images from dataset!

    global composite_labels 

    assert dtype == "segmentation"
    
    images = [x for x in glob.glob(os.path.join(folder_path, path, "*")) if not os.path.isdir(x)]
    
    if sample_dataset:
        images = images[:60]

    anns = ['/'.join(x.split('/')[:-1] + ["json", x.replace(".jpg", "__labels.json").split('/')[-1]]) for x in images]

    segmentation_data, delete_idxs = {}, []
    for idx, (img, ann) in enumerate(zip(images, anns)):
        if not os.path.exists(ann):
            delete_idxs.append(idx)
            continue
        # quicker to catch before training starts and delete all idx files from directory
        """
        try:
            img_arr = cv2.imread(img)
            arr, counts = np.unique(img_arr, return_counts=True)
            if len(arr) == 2:
                img = 1 / 0. 
            if img_arr is None:
                img = 1 / 0.
        except Exception:
            delete_idxs.append(idx)
            continue
        """
        
        img_key = img.split('/')[-1].split('.jpg')[0]
        segmentation_data[img_key] = {"image": img, "segment": ann}

    for idx in reversed(delete_idxs):
        del images[idx]
        del anns[idx]
    
    dataset = DeepFishDataset(segmentation_data, img_shape, min_segment_positivity_ratio, sample_dataset=sample_dataset, organs=organs, augment_flag=augment_flag)
    print ("Using %d labeled images from dataset: %s!" % (len(dataset), "Segmentation dataset: %s" % path))
    
    return dataset

if __name__ == "__main__":
    
    data_dir = "/home/hans/data/"
    dataset_dir = "Deep Fish/" 

    dset = get_deepfish_segclsloc_data(dtype="segmentation", path=dataset_dir, folder_path=data_dir, 
            img_shape=256, min_segment_positivity_ratio=0.05, sample_dataset=False, organs=["whole_body"])

    for img, seg, fpath in dset:
        cv2.imshow('f', img.transpose((1,2,0)))
        cv2.imshow('g', seg[0])
        key = cv2.waitKey()

