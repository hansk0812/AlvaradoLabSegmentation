import sys 

import torch
from torch import nn
from torchsummary import summary
from torchvision.models import __file__, __dict__, efficientnet_v2_s, EfficientNet_V2_S_Weights

weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
model = efficientnet_v2_s(weights)

#torch.save(model, "efficientnet.pth")
#print (summary(model, (3,256,256))); exit()

class UNet(nn.Module):
    
    def __init__(self, backbone_efficientnet):
        super().__init__()        
        self.net = backbone_efficientnet

    def forward(self, x):
                
        for layer in self.net._modules:
            if layer == "features":
                for index, layer_component in enumerate(self.net._modules[layer]._modules):
                    print (self.net._modules[layer]._modules[layer_component])
                    
                    x = self.net._modules[layer]._modules[layer_component](x)
                    
                    if index == 2-1:
                        print ('2', x.shape)
                    elif index == 3-1:
                        print ('3', x.shape)
                    
                    print ("\n\n" + '.'*50 + "\n\n")
        return x

class EfficientNetDeconv(nn.Module):
    
    def __init__(self, features, input_shape = (1280, 8, 8)):
        super().__init__()
        self.features = features
        self.input_shape = input_shape

    def forward(self, x):
        return x

net = UNet(model)
#net = net.cuda()
print (net.forward(torch.ones((1,3,256,256))).shape)

with open('efficientnet.txt', 'w') as sys.stdout:
    summary(net, (3, 256, 256))
