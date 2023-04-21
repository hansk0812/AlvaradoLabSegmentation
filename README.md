# AlvaradoLabSegmentation
Dataset Tools:

Use Python script to separate cropped segmentation parts and use it for semantic segmentation labels
`python -m ecology_semantic_segmentation.dataset.utils` 

U-Net based Semantic Segmentation

Available Backbones:

1. VGG

To train a VGG based U-Net with depth 256 i.e. 64, 128, 256 channels and image size 3x128x128, use:
`IMGSIZE=128 MAXCHANNELS=256 python -m ecology_semantic_segmentation.train --batch_size 39`
Configure `batch_size` argument to fully consume your GPU memory during training!

2. EfficientNet
