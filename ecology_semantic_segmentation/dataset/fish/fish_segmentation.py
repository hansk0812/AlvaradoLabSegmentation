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
from ..bbox_masks_problem import remove_islands_in_segment_gt

def imread(file_path):
    
    if ".arw" not in file_path.lower():
        return cv2.imread(file_path)
    else:
        img = rawpy.imread(file_path) 
        img = img.postprocess() 
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 

class SegmentationDataset(Dataset):

    def __init__(self, segmentation_data, img_shape, min_segment_positivity_ratio, sample_dataset = True, organs=None): 
        
#        if sample_dataset:
#            segmentation_data = {key: segmentation_data[key] for key in list(segmentation_data)[:60]}
        
        # Ensure all files contribute to data wrt organs
        if organs is None:
            test_organs = composite_labels
        else:
            test_organs = organs

        removable_keys = []
        for key in segmentation_data:
            
            ctx = 0
            for organ in test_organs:
                try:
                    imread(segmentation_data[key]["segments"][organ])
                    ctx += 1
                except Exception:
                    continue
            
            if ctx == 0:
                removable_keys.append(key)
        
        for key in removable_keys:
            del segmentation_data[key]

        self.segmentation_data = segmentation_data
        self.segmentation_keys = list(segmentation_data.keys())
        self.img_shape = img_shape 
        self.min_segment_positivity_ratio = min_segment_positivity_ratio
        self.organs = organs
        
        if not organs is None:
            self.label_indices = sorted([composite_labels.index(organ) for organ in organs])
        else:
            self.label_indices = list(range(len(composite_labels)))

        self.set_augment_flag(True)

    def __len__(self):
        return len(self.segmentation_keys)
    
    def set_augment_flag(self, flag):
        self.augment_flag = flag
    
    def use_bbox_for_mask(self, img, mask, bbox_file="output.json"):
        
        with open(bbox_file, 'r') as f:
            bboxes_dict = json.load(f)
        
        mask = cv2.resize(mask, (256,256)) 

    def __getitem__(self, idx):
        
        data_key = self.segmentation_keys[idx]
        
        image_path, segments_paths = self.segmentation_data[data_key]["image"], self.segmentation_data[data_key]["segments"]
        image = imread(image_path)
        image = cv2.resize(image, (self.img_shape, self.img_shape))
        
        num_segments = len(composite_labels) if self.organs is None else len(self.organs)
        segment_array = np.zeros((self.img_shape, self.img_shape, num_segments)) 
        
        for organ_index in self.label_indices:
            
            try:
                organ = composite_labels[organ_index]
                segment = imread(segments_paths[organ])
            
                segment = cv2.resize(segment, (self.img_shape, self.img_shape))

                segment = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
                
                segment = cv2.bitwise_not(segment)
                segment[segment>0] = 1
                
                segment = remove_islands_in_segment_gt(segment)
                
                # Decide using this image: Machine learning training set (copy)/photos 1.30.2019/original image/f132C.png
                #SEGMENT_THRESHOLD = 225
                #segment[segment > SEGMENT_THRESHOLD] = 0
                #segment[segment != 0] = 255
                
                area_of_segment = segment.sum() / 255.0
                
                if area_of_segment * 255 < (self.min_segment_positivity_ratio * self.img_shape * self.img_shape):
                    #TODO: Ignore labels
                    segment.fill(0) # (-1)
                
                segment_array[:, :, organ_index] = segment 
            
            except Exception:
                traceback.print_exc()
                segment_array[:, :, organ_index].fill(-1) 
        
        if self.augment_flag:
            image, segment_array = augment_fn(image, segment_array)
        
        return image.transpose((2,0,1)).astype(np.float32), segment_array.transpose((2,0,1)).astype(np.float32), image_path

def get_ml_training_set_data(dtype, path, folder_path, img_shape, min_segment_positivity_ratio, sample_dataset=True, organs=None):
    
    #TODO: 9 missing images from dataset!

    global composite_labels 

    assert dtype == "segmentation/composite"
    
    folders = [x for x in glob.glob(os.path.join(folder_path, path, "*")) \
                if os.path.isdir(x)]
    
    data = {}
    for directory in folders:
        
        dir_folders = glob.glob(os.path.join(directory, "*"))
        
        images = glob.glob(os.path.join(directory, 'original image/*'))
        
        if sample_dataset:
            images = images[:20]

        for image_path in images:
            fname = "/".join(image_path.split('/')[-3:])
            search_key = '.'.join(fname.split('/')[-1].split('.')[:-1])
            data_index = os.path.join(directory.split('/')[-1], search_key)
            
            segments_path = glob.glob(os.path.join(directory, "*", search_key + "*"))
            segments = [x.split('/')[-2] for x in segments_path]
            segments.remove("original image")
            
            if not os.path.exists(image_path):
                #TODO print (image_path)
                continue

            segment_paths = {}
            for organ in segments:
                ann_paths = glob.glob(os.path.join(directory, organ, search_key + "*")) 
                
                organ = organ.replace(" ", "_")
                if not organ in composite_labels:
                    composite_labels.append(organ)

                if len(ann_paths) == 1:
                    if os.path.exists(ann_paths[0]):
                        segment_paths[organ] = ann_paths[0]  
            
            if len(segment_paths) > 0:

                try:
                    img = cv2.imread(image_path)
                    assert not img is None

                    data[data_index] = {"image": image_path, \
                                        "segments": segment_paths}

                except Exception:
                    pass

    dataset = SegmentationDataset(data, img_shape, min_segment_positivity_ratio, sample_dataset=sample_dataset, organs=organs)
    print ("Using %d labeled images from dataset: %s!" % (len(dataset), "Segmentation dataset: %s" % path))
    
    return dataset

if __name__ == "__main__":

    data_dir = os.path.join(os.path.abspath("."), 'ecology_semantic_segmentation', 'dataset')
    
    dset = get_ml_training_set_data(dtype="segmentation/composite", path="gray", folder_path=data_dir, img_shape=256, min_segment_positivity_ratio=0.05, sample_dataset=False, organs=["whole_body"])

    for img, seg, fpath in dset:
        print (img.shape, seg.shape, fpath)
