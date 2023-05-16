import traceback
import json
import os

metadata_file = [os.path.join(os.path.dirname(__file__), x) \
                    for x in os.listdir(os.path.dirname(__file__)) \
                    if x.endswith(".json")][0]
try:
    print ("Using dataset configuration from: ", metadata_file)
    with open(metadata_file, "r") as f:
        datasets_metadata = json.load(f)
except Exception:
    datasets_metadata = None
    print ("datasets_metadata.json file unavailable!")
    traceback.print_exc()

from .dataset.fish import fish_train_dataset, fish_val_dataset, fish_test_dataset

#TODO: Change model name across all train_ and test_files
from .model import vgg_unet
unet_model = vgg_unet

import torch
binary_cross_entropy = torch.nn.BCEWithLogitsLoss()

__all__ = ["fish_train_dataset", "fish_val_dataset", "fish_test_dataset", "datasets_metadata", "binary_cross_entropy"]
