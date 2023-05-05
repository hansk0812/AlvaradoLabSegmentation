# AlvaradoLabSegmentation
Dataset Tools:

Use Python script to separate cropped segmentation parts and use it for semantic segmentation labels
`python -m ecology_semantic_segmentation.dataset.utils` 

After finding too many such cases (~150 manually fixed examples), I have FINALLY decided to write a program to solve the problem!
`python -m ecology_semantic_segmentation.dataset.bbox_masks_problem`

U-Net based Semantic Segmentation

Using U-Net models from `segmentation_models_pytorch`
`
pip install segmentation_models_pytorch 
`

Datasets:
  ML Training Set - AlvaradoLab annotated data - composite segmentation
  Cichlid Collection - AlvaradoLab annotated data - composite segmentation
  SUIM - semantic segmentation with >1 fish per image
  Deep Fish - fish_tray_images - Accurately labeled sharp masks - Semantic segmentation with >1 fish per image

Available Backbones:
  Resnet34

