from .. import colors, CPARTS, DATASET_TYPES
from ..visualize_composite_labels import display_composite_annotations

dataset_splits = {"train": 0.85, "val": 0.05, "test": 0.1}
composite_labels = []

from .fish_dataset import FishDataset, FishSubsetDataset

import os
def get_env_variable(name, default_value):
    try:
        return os.environ[name]
    except Exception:
        return default_value

SAMPLE_DATASET = bool(get_env_variable("SAMPLE", False))
IMG_SIZE = int(get_env_variable("IMGSIZE", 256))
ORGANS = [x for x in get_env_variable("ORGANS", "whole_body").split(',')]

# Deep Supervision implementation pending!
deepsupervision = False

fish_train_dataset = FishDataset(dataset_type=["segmentation/composite"], 
                                 img_shape=IMG_SIZE, 
                                 sample_dataset=SAMPLE_DATASET,
                                 deepsupervision=deepsupervision,
                                 organs=ORGANS)
print ("train dataset: %d images" % len(fish_train_dataset))

fish_val_datasets, val_cumsum_lengths, \
fish_test_datasets, test_cumsum_lengths = fish_train_dataset.return_val_test_datasets()

fish_val_dataset = FishSubsetDataset(fish_val_datasets, val_cumsum_lengths, deepsupervision=deepsupervision) 
[dataset.dataset.set_augment_flag(False) for dataset in fish_val_dataset.datasets]
print ("val dataset: %d images" % len(fish_val_dataset))

fish_test_dataset = FishSubsetDataset(fish_test_datasets, test_cumsum_lengths, deepsupervision=deepsupervision) 
[dataset.dataset.set_augment_flag(False) for dataset in fish_test_dataset.datasets]
print ("test dataset: %d images" % len(fish_test_dataset))

dataset_subsets = ["fish_train_dataset", "fish_val_dataset", "fish_test_dataset"]

#for data in fish_test_dataset:
#    image, segment = data
#    display_composite_annotations(image, segment, composite_labels, fish_test_dataset.min_segment_positivity_ratio)
#exit()

__all__ = [*dataset_subsets, "datasets_metadata", "composite_labels", "test_set_ratio", 
            "visualize_composite_labels", "colors", "CPARTS", "DATASET_TYPES"]
