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

from .fish_coco_annotator import get_alvaradolab_data
from .fish_segmentation import get_ml_training_set_data

import traceback

import albumentations as A
from albumentations.augmentations.geometric.functional import rotate
from albumentations.augmentations.blur.functional import gaussian_blur, defocus, zoom_blur

try:
    from imgaug import augmenters as iaa
except ImportError:
    import imgaug.imgaug.augmenters as iaa 

from albumentations.augmentations.transforms import Sharpen
#from albumentations.imgaug.stubs import IAASharpen

#TODO ChainDataset: In-memory dataset seemed faster

class FishDataset(Dataset):

    def __init__(self, dataset_type="segmentation", config_file = "fish_metadata.json", 
                    img_shape = 256, min_segment_positivity_ratio=0.0075, organs=["whole_body"],
                    sample_dataset=True): 
        # min_segment_positivity_ratio is around 0.009 - 0.011 for eye (the smallest part)
        
        global composite_labels

        assert dataset_type in DATASET_TYPES
        
        with open(config_file, "r") as f:
            datasets_metadata = json.load(f)
        
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
    
    def augment_image(self, img, segments):
        
        transform = A.Compose([A.OneOf([
#            A.RandomCrop(width=256, height=256),
#            A.RandomCrop(width=128, height=128),
#            A.RandomCrop(width=64, height=64),
#            ], p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45)
                ], p=0.2),
            A.Perspective(scale=(0.05, 0.1)), 
            A.Affine(scale=(0.3, 0.3), translate_percent=[0.3, 0.3], shear= np.random.rand() * 0.3),
            A.OpticalDistortion(distort_limit = .85, shift_limit = .85),
        ], p=0.25)])
        transformation = transform(image=img, mask=segments)
        img, segments = transformation["image"].transpose((2,0,1)), transformation["mask"].transpose((2,0,1))

        if np.random.rand() < 0.5:
            angle = np.random.randint(75) 
            img = rotate(img, angle)
            segments = np.array([rotate(x, angle) for x in segments])
        
#        if np.random.rand() < 0.2:
#            ksize= 3
#            img = gaussian_blur(img.astype(np.uint8), ksize=ksize)
#        elif 0.2 < np.random.rand() < 0.4:
        if np.random.rand() < 0.2:
            img = defocus(img, radius=5, alias_blur=2)

#        I = Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0))
#        if np.random.rand() > 0.6:
#            img = I(image=img)["image"]

        print ('aug', img.shape, segments.shape)
        return img, segments

    def __len__(self):
        return self.dataset_cumsum_lengths[-1]

    def __getitem__(self, idx):
        
        current_dataset_id = 0
        while idx >= self.dataset_cumsum_lengths[current_dataset_id]:
            current_dataset_id += 1
        if current_dataset_id > 0:
            data_index = idx - self.dataset_cumsum_lengths[current_dataset_id-1]       
        else:
            data_index = idx

        dataset = self.datasets[current_dataset_id]
        
        image, segment, filename = dataset[data_index]
        #image, segment = self.augment_image(image, segment)
        #image, segment = image.transpose((2,0,1)), segment.transpose((2,0,1))

        return image / 255.0, segment / 255.0, filename

class FishSubsetDataset(Dataset):
    
    def __init__(self, datasets, cumsum_lengths, min_segment_positivity_ratio=0.0075):
        
        self.min_segment_positivity_ratio = min_segment_positivity_ratio
        self.datasets = datasets
        self.dataset_cumsum_lengths = cumsum_lengths

    def __len__(self):
        return self.dataset_cumsum_lengths[-1]

    def __getitem__(self, idx):
    
        #TODO: Iterate from given idx using while loop - shuffle=False preferred for val and test sets
        current_dataset_id = 0
        while idx >= self.dataset_cumsum_lengths[current_dataset_id]:
            current_dataset_id += 1
        dataset = self.datasets[current_dataset_id]
        
        if current_dataset_id == 0:
            data_index = idx
        else:
            data_index = idx - self.dataset_cumsum_lengths[current_dataset_id-1]
        
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
    
    print ("HERE", dataset.__getitem__(0))
    for data in dataset:
        _, seg, _ = data
        seg = seg.transpose((1,2,0)).astype(np.uint8)*255
        cv2.imwrite('g.png', seg)

    val_datasets, val_cumsum_lengths, \
    test_datasets, test_cumsum_lengths = dataset.return_val_test_datasets()

    valdataset = FishSubsetDataset(val_datasets, val_cumsum_lengths) 
    print ("val dataset: %d images" % len(valdataset))
    testdataset = FishSubsetDataset(test_datasets, test_cumsum_lengths) 
    print ("test dataset: %d images" % len(testdataset))

