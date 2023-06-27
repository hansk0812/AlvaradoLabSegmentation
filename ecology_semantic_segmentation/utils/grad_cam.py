from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from . import fish_train_dataset, fish_val_dataset, fish_test_dataset

from .dataset.fish import get_env_variable
exp_models = get_env_variable("EXPTNAME", default_value="deeplabv3_leader")
ORGANS = get_env_variable("ORGANS", default_value="whole_body,ventral_side,dorsal_side").split(',')

import segmentation_models_pytorch as smp
model = smp.DeepLabV3Plus(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=len(ORGANS),                      # model output channels (number of classes in your dataset)
        )
from .train_multiclass import load_recent_model

from matplotlib import pyplot as plt

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("epoch", help="Epoch to load model from", type=int)
args = ap.parse_args()

models_dir = os.path.join("models", exp_models)

load_recent_model(models_dir, model, args.epoch)
#print (model.decoder)

dataloader = DataLoader(fish_test_dataset, shuffle=False, batch_size=2, num_workers=0)

target_layers = [model.decoder]

for img, seg, _ in dataloader:

    input_tensor = img # Create an input tensor image for your model..
    rgb_img = input_tensor[0].numpy().transpose((1,2,0))
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.

    targets = [SemanticSegmentationTarget(idx, seg[:,idx].numpy()) for idx in range(seg.shape[1])]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    plt.imshow(visualization)
    plt.show()
