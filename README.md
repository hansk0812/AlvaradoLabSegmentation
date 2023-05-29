# AlvaradoLabSegmentation
#### Dataset Tools:

Use Python script to separate cropped segmentation parts and use it for semantic segmentation labels
`python -m ecology_semantic_segmentation.dataset.utils` 

After finding too many such cases (~150 manually fixed examples), I have FINALLY decided to write a program to solve the problem!
`python -m ecology_semantic_segmentation.dataset.bbox_masks_problem`

#### U-Net based Semantic Segmentation

Using U-Net models from `segmentation_models_pytorch`
`pip install segmentation_models_pytorch `

#### Datasets:
  ML Training Set - AlvaradoLab annotated data - composite segmentation
  Cichlid Collection - AlvaradoLab annotated data - composite segmentation
  SUIM - semantic segmentation with >1 fish per image
  Deep Fish - fish_tray_images - Accurately labeled sharp masks - Semantic segmentation with >1 fish per image

##### Available Pre-trained Backbones:
  Resnet34
  Resnet50
  DeepLabv3Plus

Link to trained models: https://drive.google.com/drive/folders/1jrUJtpxR8WvUf3td9nWXMIEdPmsgK_gp?usp=share_link

#### Training script:

Use `SAMPLE=1`, `SAMPLE=0`, `IMG_SIZE=256`, `ORGANS="whole_body"` flags to control training and debugging training code
Using batch sizes as multiples of 9 gives most efficient use of GPU space

`ORGANS=whole_body,ventral_side,dorsal_side python -m ecology_semantic_segmentation.train_multiclass --batch_size 54`
`python -m ecology_semantic_segmentation.train --batch_size 54`

Learning Rate Scheduling for Adam: 0.0003 to start, re-define for every checkpoint resume manually using script parameter 

#### Testing Script:

Uses Dice score for accuracy
`python -m ecology_semantic_segmentation.test --models <MODELDIR>`

#TODO: DEEPSUPERVISION CODE!
