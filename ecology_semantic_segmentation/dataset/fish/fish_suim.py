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

#from .fish_segmentation import imread

class SUIMDataset(Dataset):

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
        
        image_path, segments_paths = self.segmentation_data[data_key]["image"], self.segmentation_data[data_key]["segments"]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.img_shape, self.img_shape))
        
        num_segments = 1 # whole_body only
        segment_array = np.zeros((self.img_shape, self.img_shape, num_segments)) 
        
        organ = self.organs[num_segments-1]
        segment = cv2.imread(segments_paths[num_segments-1])
        segment = cv2.inRange(cv2.cvtColor(segment, cv2.COLOR_BGR2HSV), (20, 100, 100), (30, 255, 255), 255)
        segment = cv2.resize(segment, (self.img_shape, self.img_shape))
        
        segment_array[:,:,0] = segment

        #segment = remove_islands_in_segment_gt(segment)
        
        # Decide using this image: Machine learning training set (copy)/photos 1.30.2019/original image/f132C.png
        #SEGMENT_THRESHOLD = 225
        #segment[segment > SEGMENT_THRESHOLD] = 0
        #segment[segment != 0] = 255
        
        if self.augment_flag:
            image, segment_array = augment_fn(image, segment_array)
        
        return image.transpose((2,0,1)).astype(np.float32) / 255.0, segment_array.transpose((2,0,1)).astype(np.float32) / 255.0, image_path

def get_suim_data(dtype, path, folder_path, img_shape, min_segment_positivity_ratio, 
                                sample_dataset=True, organs=["whole_body"], bbox_dir=None ,augment_flag = True):
    
    #TODO: 9 missing images from dataset!

    global composite_labels 

    assert dtype == "segmentation"
    
    images = glob.glob(os.path.join(folder_path, path, "*", '*'))
    segmentation_data = {}
    for imgpath in images:
        fname = '.'.join(imgpath.split('/')[-1].split('.')[:-1])
        if "/images/" in imgpath:
            if fname in segmentation_data:
                segmentation_data[fname]["image"] = imgpath
            else:
                segmentation_data[fname] = {"image": imgpath, "segments": []}
        else:
            if fname in segmentation_data:
                segmentation_data[fname]["segments"].append(imgpath)
            else:
                segmentation_data[fname] = {"image": None, "segments": [imgpath]}
    
    
    removable_keys = []
    for file_id in segmentation_data.keys():
        img, segs = segmentation_data[file_id]["image"], segmentation_data[file_id]["segments"]
        
        try:
            assert len(segs) == 1, "Using whole_body annotations only in SUIM dataset"
            
            img = cv2.imread(img)
            segs = cv2.imread(segs[0])

        except Exception:
            removable_keys.append(file_id)

    for key in removable_keys:
        del segmentation_data[key]

    dataset = SUIMDataset(segmentation_data, img_shape, min_segment_positivity_ratio, sample_dataset=sample_dataset, organs=organs, augment_flag=augment_flag)
    print ("Using %d labeled images from dataset: %s!" % (len(dataset), "Segmentation dataset: %s" % path))
    
    return dataset

if __name__ == "__main__":
    
    data_dir = "/home/hans/data/"
    dataset_dir = "SUIM/SUIM/train_val/" 
    
    print ("dtype=\"segmentation\"", "path=", dataset_dir, "folder_path=", data_dir, 
            "img_shape=256", "min_segment_positivity_ratio=0.05", "sample_dataset=True", "organs=[\"whole_body\"]")
    dset = get_suim_data(dtype="segmentation", path=dataset_dir, folder_path=data_dir, 
            img_shape=256, min_segment_positivity_ratio=0.05, sample_dataset=True, organs=["whole_body"])

    for img, seg, fpath in dset:
        print (img.min(), img.max())
        cv2.imshow('f', img.transpose((1,2,0)))
        cv2.imshow('g', seg[0])
        cv2.waitKey()

        print (img.shape, seg.shape, fpath)
