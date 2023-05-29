import argparse

import os
import glob

import json
import cv2
import numpy as np

import torch 
from torch.utils.data import Dataset, DataLoader

from . import display_composite_annotations
from . import colors, CPARTS, DATASET_TYPES
from . import dataset_splits

from .. import datasets_metadata

# All datasets have get_%s_data
from .fish_coco_annotator import get_alvaradolab_data
from .fish_segmentation import get_ml_training_set_data
from .fish_suim import get_suim_data
from .fish_deepfish_segment import get_deepfish_segclsloc_data

import traceback

#TODO ChainDataset: In-memory dataset seemed faster

class FishDataset(Dataset):

    def __init__(self, dataset_type=["segmentation/composite"], img_shape = 256, min_segment_positivity_ratio=0.0075, 
                 organs=["whole_body"], sample_dataset=True, deepsupervision=False): 
        # min_segment_positivity_ratio is around 0.009 - 0.011 for eye (the smallest part)

        assert all([x in DATASET_TYPES for x in dataset_type]), ",".join([x + str(x in DATASET_TYPES) for x in dataset_type])
        
        self.organs = organs

        self.folder_path = datasets_metadata["folder_path"] 
        datasets = datasets_metadata["datasets"] 
       
        self.min_segment_positivity_ratio = min_segment_positivity_ratio
        self.xy_pairs = []

        # Accepts single type of data only
        datasets = list([x for x in datasets if x["type"] in dataset_type])
        
        self.curated_images_count, self.dataset_generators = 0, []
        
        self.get_alvaradolab_data = get_alvaradolab_data
        self.get_ml_training_set_data = get_ml_training_set_data
        self.get_suim_data = get_suim_data
        self.get_deepfish_segclsloc_data = get_deepfish_segclsloc_data

        self.datasets, self.dataset_cumsum_lengths = [], []
        self.val_datasets, self.test_datasets = [], []

        for data in datasets:
            
            dataset_method = "get_%s_data" % data["name"]

            if "bbox_dependent_images_path" in data:
                bbox_dir = data["bbox_dependent_images_path"]
            else:
                bbox_dir = None
            
            try:
                dataset = getattr(self, dataset_method)(data["type"], data["folder"],
                                                        self.folder_path, 
                                                        img_shape, min_segment_positivity_ratio,
                                                        organs = organs,
                                                        sample_dataset = sample_dataset,
                                                        bbox_dir = None, #bbox_dir)
                                                        augment_flag = True) 
                
                # create train, val or test sets
                num_samples = {"train": [0, int(len(dataset) * dataset_splits["train"])]}
                num_samples["val"] = [num_samples["train"][1], num_samples["train"][1] + int(len(dataset) * dataset_splits["val"])] 
                num_samples["test"] = [num_samples["val"][1], len(dataset)]
                
                indices = range(*num_samples["train"])
                self.datasets.append(torch.utils.data.Subset(dataset, indices))
               
                if len(self.dataset_cumsum_lengths) == 0:
                    self.dataset_cumsum_lengths.append(len(indices))
                else:
                    self.dataset_cumsum_lengths.append(self.dataset_cumsum_lengths[-1] + len(indices))

                indices = range(*num_samples["val"])
                self.val_datasets.append(torch.utils.data.Subset(dataset, indices))
                indices = range(*num_samples["test"])
                self.test_datasets.append(torch.utils.data.Subset(dataset, indices))
 
            except Exception as e:
                traceback.print_exc()
                print ("Write generator function for dataset: %s ;" % dataset_method, e)
    
        self.deepsupervision = deepsupervision

    def return_val_test_datasets(self):
        
        val_cumsum_lengths, test_cumsum_lengths = [], []
        for dataset in self.val_datasets:
            if len(val_cumsum_lengths) == 0:
                val_cumsum_lengths.append(len(dataset))
            else:
                val_cumsum_lengths.append(val_cumsum_lengths[-1] + len(dataset))
        for dataset in self.test_datasets:
            if len(test_cumsum_lengths) == 0:
                test_cumsum_lengths.append(len(dataset))
            else:
                test_cumsum_lengths.append(test_cumsum_lengths[-1] + len(dataset))

        return self.val_datasets, val_cumsum_lengths, \
               self.test_datasets, test_cumsum_lengths

    def get_relative_ratios(self):

        ratios = [0 for _ in range(len(self.organs))]
        for _, segment, _ in self:
            for organ_index in range(segment.shape[-3]):
                gt = segment[organ_index]
                ratios[organ_index] += np.sum(gt)
        ratios = np.array(ratios) / len(self)
        ratios /= np.max(ratios)
        
        return ratios

    def __len__(self):
        return self.dataset_cumsum_lengths[-1]
    
    def return_descending_order_sets(self, img, ann):
        # Preferred order: easiest to segment organ as ann[0] --> hardest to segment as ann[-1]
        for idx in range(ann.shape[0]-1):
            ann[idx] = sum(x for x in ann[idx+1:])
        ann[ann>1] = 1
        return ann

    def __getitem__(self, idx):
        
        current_dataset_id = 0
        while idx >= self.dataset_cumsum_lengths[current_dataset_id]:
            current_dataset_id += 1
        
        dataset = self.datasets[current_dataset_id]

        data_index = idx if current_dataset_id == 0 else \
                idx - self.dataset_cumsum_lengths[current_dataset_id-1]        
        
        assert data_index < len(dataset), \
                "%d > %d ; Dataset %d / %d" % (data_index, len(dataset), current_dataset_id, len(self.dataset_cumsum_lengths))
        image, segment, filename = dataset[data_index]
    
        segment[segment > 0] = 1
        if self.deepsupervision:
            small_segments = [np.expand_dims(cv2.resize(segment[0], (idx, idx)), axis=0) for idx in [128, 64, 32, 16, 8]]
            segment = [segment] + small_segments
        else:
            if segment.max() > 1:
                segment = segment / 255.0
        if image.max() > 1:
            image = image / 255.0
        
        return image, segment, filename

class FishSubsetDataset(Dataset):
    
    def __init__(self, datasets, cumsum_lengths, min_segment_positivity_ratio=0.0075, deepsupervision=True):
        
        self.min_segment_positivity_ratio = min_segment_positivity_ratio
        self.datasets = datasets
        self.dataset_cumsum_lengths = cumsum_lengths

        self.deepsupervision = deepsupervision
    
    def __len__(self):
        return self.dataset_cumsum_lengths[-1]

    def __getitem__(self, idx):
        
        current_dataset_id, data_index = 0, idx
        while idx >= self.dataset_cumsum_lengths[current_dataset_id]:
            current_dataset_id += 1
            data_index = idx - self.dataset_cumsum_lengths[current_dataset_id-1]
        dataset = self.datasets[current_dataset_id]
        
        image, segment, filename = dataset[data_index]

        segment[segment > 0] = 1
        if self.deepsupervision:
            small_segments = [np.expand_dims(cv2.resize(segment[0], (idx, idx)), axis=0) for idx in [128, 64, 32, 16, 8]]
            segment = [segment] + small_segments
        
        return image, segment, filename

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--visualize", default="alvaradolab", help="Flag to visualize composite labels")
    args = ap.parse_args()

    dataset = FishDataset(dataset_type=["segmentation/composite"], sample_dataset=False, organs=["whole_body", "ventral_side", "dorsal_side"]) #"segmentation", 
    print ("train dataset: %d images" % len(dataset))

    val_datasets, val_cumsum_lengths, \
    test_datasets, test_cumsum_lengths = dataset.return_val_test_datasets()

    valdataset = FishSubsetDataset(val_datasets, val_cumsum_lengths) 
    print ("val dataset: %d images" % len(valdataset))
    
    print (dataset.get_relative_ratios())
    exit()

    for img, seg, fname in dataset:
        img = img.transpose((1,2,0))*255 
        cv2.imshow("f", img.astype(np.uint8))
        cv2.imshow("g", (seg[0]*255).astype(np.uint8))
        cv2.imshow("h", (seg[1]*255).astype(np.uint8))
        cv2.imshow("i", (seg[2]*255).astype(np.uint8))
        print (fname)
        cv2.waitKey()

    testdataset = FishSubsetDataset(test_datasets, test_cumsum_lengths) 
    print ("test dataset: %d images" % len(testdataset))

