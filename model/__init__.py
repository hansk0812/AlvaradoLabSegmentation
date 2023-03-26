from .vgg import VGGUNet
from .vgg import vgg19_bn as VGGClassifier, VGG19_BN_Weights

vgg_classifier = VGGClassifier(weights=VGG19_BN_Weights.DEFAULT)
vgg_unet = VGGUNet(vgg_classifier)

__all__ = ["vgg_unet"]
