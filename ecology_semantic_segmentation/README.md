# AlvaradoLabSegmentation
U-Net based Semantic Segmentation

## Available Datasets:

##### segmentation/composite
- `ecology_semantic_segmentation/dataset/fish/fish_coco_annotator.py`
- `ecology_semantic_segmentation/dataset/fish/fish_segmentation.py`

##### segmentation
- `ecology_semantic_segmentation/dataset/fish/fish_suim.py`
- `ecology_semantic_segmentation/dataset/fish/fish_deepfish_segment.py`

## Dataset Toolkit Scripts
- Re-alignment of segmentation masks based on bounding box based cropping of masks: `ecology_semantic_segmentation/dataset/utils/bbox_masks_problem.py`
- Visualize composite segmentation labels: `ecology_semantic_segmentation/dataset/visualize_composite_labels.py`
- Dataset augmentation methods: `ecology_semantic_segmentation/dataset/augment.py`
- Pick HSV color ranges based on mouse click on CV2 image: `ecology_semantic_segmentation/dataset/utils/hsv_picker.py`
- Create new folder dataset for ML training set pairs with bbox masks problem: `ecology_semantic_segmentation/dataset/utils/manual_bbox_identification.py`

##### Dataset resources
- Colors for composite segmentation visualization: `ecology_semantic_segmentation/dataset/resources/color_constants.py`
- Classnames used in COCO Dataset Generator GUI tool: `ecology_semantic_segmentation/dataset/resources/classnames/alvaradolab.txt`
- Color palette HSV ranges: `ecology_semantic_segmentation/dataset/resources/color_palette.txt`
- Manually extracted directory structure from analysing ML training set: `ecology_semantic_segmentation/dataset/resources/composite_bboxes.py`
- Color palette with 24 colors used for experiment: `ecology_semantic_segmentation/dataset/resources/palette.png`

## Available Backbones:

`pip install segmentation_models_pytorch`

- Resnet34
- Resnet50

## Available organs:
- `whole_body` single organ models
- `ventral_side` single organ models
- `whole_body,ventral_side,dorsal_side` multi-organ models (loss based stability study ongoing)
- `whole_body,ventral_side,dorsal_side` superset models
    ##### Drop in performance for ventral_side UNION dorsal_side models - suspected federated knowledge distillation
    ##### HOPE: time doesn't make it work well

## Training specifics and model changes
- `ecology_semantic_segmentation/train.py`
- `ecology_semantic_segmentation/test.py`


