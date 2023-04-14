import torch
from torch import nn 
from torch.nn import functional as F
from torchsummary import summary

from torchvision.models import vgg19_bn, VGG19_BN_Weights
from .dropout import StochasticDropout

class DeconvNormActivation(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups = 1, padding=None, num_blocks=2, bias=True, dropout_p=0.05):
        
        super().__init__()
        
        self.block = nn.ModuleList([])

        for idx in range(num_blocks):
            deconv = nn.ConvTranspose2d(in_channels = in_channels if idx == 0 else out_channels, 
                                        out_channels = out_channels, 
                                        kernel_size = kernel_size, 
                                        stride = stride,
                                        groups = groups,
                                        padding=padding,
                                        bias=bias and idx > 0)
            batchnorm = nn.BatchNorm2d(out_channels)
            activation = nn.SiLU()

            self.block.append(deconv)
            self.block.append(batchnorm)
            self.block.append(activation)
            
            if dropout_p != 0: # and idx > 0 #Include dropouts across all deconv layers
                dropout = StochasticDropout(dropout_p)
                self.block.append(dropout)

    def forward(self, x):   
        
        for module in self.block:
            x = module(x)
        return x

class VGGUNetDecoder(nn.Module):
    
    def __init__(self, 
                 channels = [512, 512, 512, 512, 512, 256, 256, 128, 64],
                 upsample = [True, False, False, True, False, True, False, True, True],
                 num_blocks = 2,
                 out_channels = 1,
                 max_channels = 512,
                 dropout_p = 0.05,
                 dropout_min_channels = 256):
        
        super().__init__()

        assert len(channels) == len(upsample) 

        if max_channels != 512:
            channels = [x for x in channels if x <= max_channels]
            upsample = upsample[-len(channels):]
        
        channels.insert(0, channels[0])

        self.channel_blocks = nn.ModuleList([
                                DeconvNormActivation(
                                        channels[idx] if not upsample[idx] else channels[idx] + channels[idx+1], 
                                        channels[idx+1], 
                                        kernel_size=3, 
                                        stride=1, 
                                        padding=1,
                                        num_blocks = 1 if idx==0 and max_channels==512 else 2,
                                        dropout_p = dropout_p if dropout_min_channels <= channels[idx+1] else 0.) \
                                    for idx in range(len(channels)-1)])
        
        self.conv_blocks = nn.ModuleList([
                                DeconvNormActivation(
                                        channels[idx+1], 
                                        channels[idx+1], 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0,
                                        num_blocks = 2,
                                        dropout_p = dropout_p if dropout_min_channels <= channels[idx+1] else 0.) \
                                                if idx!=0 else None
                                    for idx in range(len(channels)-1)])

        self.final_conv = DeconvNormActivation(channels[-1], 1, 1, 1, padding=0, dropout_p=0.)
        self.channels = channels
        self.upsample = upsample
    
    def forward(self, x, encoder_tensors):
        
        encoder_concat_index = 0
        
        for index, (block1, block2) in \
            enumerate(zip(self.channel_blocks, self.conv_blocks)):
            
            if self.upsample[index]:
                x = F.interpolate(x, scale_factor=2)   
                x = torch.cat((encoder_tensors[encoder_concat_index], x), dim=1)
                encoder_concat_index += 1
            
            x = block1(x)
            
            if not block2 is None:
                x = block2(x)
            
        return self.final_conv(x)

class VGGUNetEncoder(nn.Module):
    
    def __init__(self, vgg_classifier, img_size=256, max_channels=512, dropout_p=0.05, dropout_min_channels=256):
        
        super().__init__()

        self.net = vgg_classifier.features
        
        dropout_net = []
        dropout_flag = False
        
        for layer in self.net:
            
            if isinstance(layer, nn.Conv2d) and layer.out_channels > max_channels:
                break

            dropout_net.append(layer)

            """
            #TODO: Does it matter whether dropout affects Max Pooling?
            if not isinstance(layer, nn.MaxPool2d):
                self.dropout_net.append(layer)
            else:
                if isinstance(self.dropout_net[-1], StochasticDropout):
                    self.dropout_net[-1] = layer
            """

            if isinstance(layer, nn.Conv2d):
                in_channels = layer.in_channels
                out_channels = layer.out_channels

                if not dropout_flag:
                    if layer.out_channels >= dropout_min_channels:
                        dropout_flag = True
            
            if dropout_flag:
                if isinstance(layer, nn.ReLU) and dropout_p != 0: # and in_channels == out_channels: #Only one block with in_channels != out_channels
                    dropout_net.append(StochasticDropout(dropout_p))
        
        self.net = nn.Sequential(*dropout_net)
        
        self.feature_size = img_size

    def forward(self, x):
        
        forward_blocks = []

        for layer in self.net:

            if isinstance(layer, nn.MaxPool2d):
                forward_blocks.append(x)
            
            x = layer(x)

        return x, list(reversed(forward_blocks))

class VGGUNet(nn.Module):
    
    def __init__(self, vgg_classifier, img_size=256, max_channels=512, dropout_p=0.05, dropout_min_channels=256):
      
        super().__init__()

        self.encoder = VGGUNetEncoder(vgg_classifier, img_size, max_channels, dropout_p, dropout_min_channels)
        self.decoder = VGGUNetDecoder(max_channels=max_channels, dropout_p=dropout_p, dropout_min_channels=dropout_min_channels)
        
        self.dropout_p, self.dropout_min_channels = dropout_p, dropout_min_channels

    def __str__(self):

        return "Using VGG U-Net with dropout_p=%.5f and dropout_min_channels=%d" % (dropout_p, dropout_min_channels)

    def forward(self, x):
        
        x, encoder_tensors = self.encoder.forward(x)
        x = self.decoder.forward(x, encoder_tensors)

        return torch.softmax(x, dim=1)

if __name__ == "__main__":
    
    net = vgg19_bn()

#    net = nn.Sequential(vgg19_bn().features)
#    print (summary(net, (3, 256, 256)))
    
#    net = VGGUNetEncoder(net)
#    output, intermediate = net.forward(torch.ones((1,3,256,256)))
#    print (output.shape, [x.shape for x in intermediate])
    
#    decoder = VGGUNetDecoder()
#    print (summary(decoder, (512, 8, 8)))
    
    net = VGGUNet(net, max_channels=128, dropout_p=0.05, dropout_min_channels=256)
    net=  net.cuda()
    print (summary(net, (3,256,256)))
