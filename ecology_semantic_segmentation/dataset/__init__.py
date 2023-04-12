from .color_constants import colors
from random import shuffle
colors = {k: colors[k] for k in colors if \
                any([x in colors for x in ["blue", "red", "cyan", "yellow", "green"]]) and \
                not any([str(x) in colors for x in range(1,5)])}
colors = list(colors.values())
shuffle(colors)

INIT = ['whole_body']

# Composite parts: 
# ventral_side seems like it needs to fully cover anal_fin and pectoral_fin from Google Search results on the topic
# dorsal_side doesn't cover dorsal fin
# operculum boundaries are outside head region
CPARTS = [['ventral_side', 'anal_fin', 'pectoral_fin'], ['dorsal_side', 'dorsal_fin'], ['head', 'eye', 'operculum']]

# Independent parts are ones without compositional overlap: whole_body contains these parts independently
INDEP = ['humeral_blotch', 'pelvic_fin', 'caudal_fin']
CPARTS.append(INDEP)
CPARTS.insert(0, INIT)

DATASET_TYPES = ["segmentation", "polygons"]
DATASET_TYPES.extend(list(map(lambda s: \
                        s + "/composite", DATASET_TYPES)))

"""
IMG_TYPES = ['jpg', 'png', 'arw']
IMG_TYPES.extend([x.upper() for x in IMG_TYPES])

# PyTorch transform functions

#JOINT_TRANSFORMS = ['CenterCrop', 'FiveCrop', 'Pad', 'RandomAffine', 'RandomCrop', 'RandomHorizontalFlip', 
#                    'RandomVerticalFlip', 'RandomResizedCrop', 'RandomRotation', 'Resize', 'TenCrop']

#IMG_TRANSFORMS = ['ColorJitter', 'Grayscale', 'RandomGrayscale']
#IMG_FTRANSFORMS = ['adjust_gamma']
"""

from .. import datasets_metadata
__all__ = ["colors", "CPARTS", "datasets_metadata"] 
