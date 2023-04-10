from .vgg import VGGUNet
from .vgg import vgg19_bn as VGGClassifier, VGG19_BN_Weights

vgg_classifier = VGGClassifier(weights=VGG19_BN_Weights.DEFAULT)
vgg_unet = VGGUNet(vgg_classifier, dropout_p=0.2, dropout_min_channels=256)

__all__ = ["vgg_unet"]
