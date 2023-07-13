from .vgg import VGGUNet
from .vgg import vgg19_bn as VGGClassifier, VGG19_BN_Weights

from .deeplabv3plus_depthwise import DeepLabV3PlusDepthwise

import os
try:
    MAX_CHANNELS = int(os.environ["MAXCHANNELS"])
except Exception:
    MAX_CHANNELS = 512

vgg_classifier = VGGClassifier(weights=VGG19_BN_Weights.DEFAULT)

# Reduced size of UNet because of single object related simplicity!
vgg_unet = VGGUNet(vgg_classifier, max_channels=MAX_CHANNELS, deepsupervision=False) #, dropout_p=0.2, dropout_min_channels=256)

__all__ = ["vgg_unet", "DeepLabV3PlusDepthwise"]
