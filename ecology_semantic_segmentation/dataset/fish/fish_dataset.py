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

from .fish_coco_annotator import get_alvaradolab_data
from .fish_segmentation import get_ml_training_set_data

import traceback

#TODO ChainDataset: In-memory dataset seemed faster

class FishDataset(Dataset):

    def __init__(self, dataset_type="segmentation", img_shape = 256, min_segment_positivity_ratio=0.0075, 
                 organs=["whole_body"], sample_dataset=True): 
        # min_segment_positivity_ratio is around 0.009 - 0.011 for eye (the smallest part)
        
        global composite_labels

        assert dataset_type in DATASET_TYPES
        
        self.folder_path = datasets_metadata["folder_path"] 
        datasets = datasets_metadata["datasets"] 
       
        self.min_segment_positivity_ratio = min_segment_positivity_ratio
        self.xy_pairs = []

        # Accepts single type of data only
        datasets = reversed(list(reversed([x for x in datasets if x["type"] == dataset_type])))
        
        self.curated_images_count, self.dataset_generators = 0, []
        
        self.get_alvaradolab_data = get_alvaradolab_data
        self.get_ml_training_set_data = get_ml_training_set_data
        
        self.datasets, self.dataset_cumsum_lengths = [], []
        self.val_datasets, self.test_datasets = [], []
        for data in datasets:
            
            dataset_method = "get_%s_data" % data["name"]
            
            try:
                dataset = getattr(self, dataset_method)(data["type"], data["folder"],
                                                        self.folder_path, 
                                                        img_shape, min_segment_positivity_ratio,
                                                        organs = organs,
                                                        sample_dataset = sample_dataset) 
         
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

    def __len__(self):
        return self.dataset_cumsum_lengths[-1]

    def __getitem__(self, idx):
        
        current_dataset_id = 0
        while idx >= self.dataset_cumsum_lengths[current_dataset_id]:
            current_dataset_id += 1
        dataset = self.datasets[current_dataset_id]
        
        data_index = idx if current_dataset_id == 0 else \
                idx - self.dataset_cumsum_lengths[current_dataset_id-1]        
        
        assert data_index < len(dataset), "%d > %d" % (data_index, len(dataset))
        image, segment, filename = dataset[data_index]
        return image / 255.0, segment / 255.0, filename

class FishSubsetDataset(Dataset):
    
    def __init__(self, datasets, cumsum_lengths, min_segment_positivity_ratio=0.0075):
        
        self.min_segment_positivity_ratio = min_segment_positivity_ratio
        self.datasets = datasets
        self.dataset_cumsum_lengths = cumsum_lengths
    
    def __len__(self):
        return self.dataset_cumsum_lengths[-1]

    def __getitem__(self, idx):
        
        current_dataset_id = 0
        while idx >= self.dataset_cumsum_lengths[current_dataset_id]:
            current_dataset_id += 1
        dataset = self.datasets[current_dataset_id]
        
        data_index = idx if current_dataset_id == 0 else \
                idx - self.dataset_cumsum_lengths[current_dataset_id-1]
        
        image, segment, filename = dataset[data_index]

        return image / 255.0, segment / 255.0, filename

if __name__ == "__main__":
   
    from . import composite_labels 
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--visualize", default="alvaradolab", help="Flag to visualize composite labels")
    ap.add_argument("--sample_dataset", action="store_true", help="Boolean to sample dataset instead of use all data")
    args = ap.parse_args()

    dataset = FishDataset(dataset_type="segmentation/composite", sample_dataset=args.sample_dataset) 
    print ("train dataset: %d images" % len(dataset))

    for img, seg, fname in dataset:
        print (fname)

    val_datasets, val_cumsum_lengths, \
    test_datasets, test_cumsum_lengths = dataset.return_val_test_datasets()

    valdataset = FishSubsetDataset(val_datasets, val_cumsum_lengths) 
    print ("val dataset: %d images" % len(valdataset))
    testdataset = FishSubsetDataset(test_datasets, test_cumsum_lengths) 
    print ("test dataset: %d images" % len(testdataset))

