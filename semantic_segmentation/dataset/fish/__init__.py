from .. import colors, CPARTS, DATASET_TYPES
from ..visualize_composite_labels import display_composite_annotations

dataset_splits = {"train": 0.85, "val": 0.05, "test": 0.1}
composite_labels = []

from .fish_dataset import FishDataset, FishSubsetDataset

fish_train_dataset = FishDataset(dataset_type="segmentation/composite", sample_dataset=True) 
print ("train dataset: %d images" % len(fish_train_dataset))

fish_val_datasets, val_cumsum_lengths, \
fish_test_datasets, test_cumsum_lengths = fish_train_dataset.return_val_test_datasets()

fish_val_dataset = FishSubsetDataset(fish_val_datasets, val_cumsum_lengths) 
print ("val dataset: %d images" % len(fish_val_dataset))

fish_test_dataset = FishSubsetDataset(fish_test_datasets, test_cumsum_lengths) 
print ("test dataset: %d images" % len(fish_test_dataset))

dataset_subsets = ["fish_train_dataset", "fish_val_dataset", "fish_test_dataset"]

#for data in fish_test_dataset:
#    image, segment = data
#    display_composite_annotations(image, segment, composite_labels, fish_test_dataset.min_segment_positivity_ratio)
#exit()

__all__ = [*dataset_subsets, "composite_labels", "test_set_ratio", "visualize_composite_labels", "colors", "CPARTS", "DATASET_TYPES"]
