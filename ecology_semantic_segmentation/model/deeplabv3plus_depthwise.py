from torch import nn
from segmentation_models_pytorch import DeepLabV3Plus

class DeepLabV3PlusDepthwise(nn.Module):

    def __init__(self, encoder_name, encoder_weights, in_channels, classes, depthwise_multiplier=5):

		self.smp_deeplab_model = smp.DeepLabV3Plus(encoder_name=encoder_name, 
											encoder_weights=encoder_weights,     
											in_channels=in_channels,                  
											classes=classes*depthwise_multiplier)

        self.last_layers = nn.Conv2d(in_channels=classes*depthwise_multiplier, 
                                    out_channels=classes, 
                                    kernel_size=3, 
                                    stride=1, 
                                    padding=0)

    def forward(self, x):

        features = self.smp_deeplab_model(x)
        output = self.last_layers(features)

        return output
