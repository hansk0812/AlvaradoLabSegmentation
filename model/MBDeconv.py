#self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
#nn.SiLU

from functools import partial

from torch import nn 

from torchvision.models.efficientnet import EfficientNet_V2_S_Weights, _efficientnet #, MBConvConfig, FusedMBConvConfig
from torchvision.ops import StochasticDepth # Bernoulli residual dropouts
from .MBDeconvConfig import MBConvConfig, FusedMBConvConfig, _MBConvConfig

from torchsummary import summary

inverted_residual_setting = [
    FusedMBConvConfig(1, 3, 1, 24, 24, 2),
#    FusedMBConvConfig(4, 3, 2, 24, 48, 4),
#    FusedMBConvConfig(4, 3, 2, 48, 64, 4),
#    MBConvConfig(4, 3, 2, 64, 128, 6),
#    MBConvConfig(6, 3, 1, 128, 160, 9),
#    MBConvConfig(6, 3, 2, 160, 256, 15),
]
last_channel = 1280

weights = None #EfficientNet_V2_S_Weights.IMAGENET1K_V1

net = _efficientnet(
    inverted_residual_setting,
    0.2,
    last_channel,
    weights,
    progress=True,
    norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
)

print (str(inverted_residual_setting[0]))
print (summary(net, (3,256,256))); 

class MBDeconv(nn.Module):
    
    def __init__(self, conv_config, stochastic_depth_prob, fused = False):
        
        #expand_ratio: 1.000000 ; kernel: 3 ; stride: 1 ; input_channels: 24 ; out_channels: 24 ; num_layers: 2
        self.conv_config = conv_config
        self.stochastic_depth_prob = stochastic_depth_prob
        
        self.use_residual_connection = conv_config.stride == 1 and conv_config.input_channels == conv_config.out_channels
        
        layers = []

        expanded_channels = conv_config.adjust_channels(conv_config.input_channels, conv_config.expand_ratio)
        
        if not fused:
            # expand
            if expanded_channels != conv_config.input_channels:
                layers.append(
                    self.deconv_block(
                        in_channels = self.conv_config.input_channels,
                        out_channels = expanded_channels,
                        kernel_size = 1,
                        stride = self.conv_config.stride
                    )
                )

            # depthwise
            layers.append(
                self.deconv_block(
                    in_channels = expanded_channels,
                    out_channels = expanded_channels,
                    kernel_size = 1,
                    stride = self.conv_config.stride,
                    groups = expanded_channels
                )
            )

            # skipping squeeze and excitation blocks for deconvolution for now!

            # project
            layers.append(
                self.deconv_block(
                    in_channels = expanded_channels,
                    out_channels = expanded_channels,
                    kernel_size = 1,
                    stride = self.conv_config.stride
                )
            )
        
        else:
            
            if expanded_channels != self.conv_config.input_channels:
                # fused expand
                layers.append(
                    self.deconv_block(
                        in_channels = self.conv_config.input_channels,
                        out_channels = expanded_channels,
                        kernel_size = self.conv_config.kernel,
                        stride = self.conv_conv_config.stride,
                    )
                )
            
            else:
                
                layers.append(
                    self.deconv_block(
                        in_channels = expanded_channels,
                        out_channels = expanded_channels,
                        kernel_size = self.conv_config.kernel,
                        stride = self.conv_conv_config.stride,
                    )
                )

        self.stochastic_depth = StochasticDepth(self.stochastic_depth_prob, "row")
    
    def deconv_block(self, in_channels, out_channels, kernel_size, stride, groups = 1):
        
        deconv = nn.ConvTranspose2d(in_channels = in_channels, 
                                    out_channels = out_channels, 
                                    kernel_size = kernel_size, 
                                    stride = stride,
                                    groups = groups)
        batchnorm = nn.BatchNorm2d(out_channels)
        activation = nn.SiLU()
        
        return nn.ModuleList([deconv, batchnorm, activation])


    def forward(self, x, concat_layers):
        
        concat_layers = torch.sum(concat_layers, dim=0)
        x = torch.cat((concat_layers, x), dim=0)
        
        y = self.block(x)

        if self.use_residual_connection:
            y = self.stochastic_depth(y)
            y += x

        return y

class DeconvNormActivation(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups = 1, padding=None):
        
        super().__init__()

        deconv = nn.ConvTranspose2d(in_channels = in_channels, 
                                    out_channels = out_channels, 
                                    kernel_size = kernel_size, 
                                    stride = stride,
                                    groups = groups,
                                    padding=padding)
        batchnorm = nn.BatchNorm2d(out_channels)
        activation = nn.SiLU()
        
        self.block = nn.ModuleList([deconv, batchnorm, activation])
    
    def forward(self, x):   
        
        for module in self.block:
            x = module(x)
        return x

# building final deconv layer
lastdeconv_output_channels = inverted_residual_setting[0].input_channels
last_block = DeconvNormActivation(lastdeconv_output_channels, 1, kernel_size=3, stride=2, padding=1)

print (summary(last_block, (24,128,128)))

exit()


MBConvConfig.adjust_channels
MBConvConfig.adjust_depth

cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)

# building first layer
firstconv_output_channels = inverted_residual_setting[0].input_channels
layers.append(
    Conv2dNormActivation(
        3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
    )
)

# expand - adjust channels and 1x1 conv
# depthwise - groups = num_channels

# squeeze and excitation
from torchvision.ops.misc import SqueezeExcitation

class SqueezeExcitation(torch.nn.Module):
     """
     This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/17
     Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``si
 
     Args:
         input_channels (int): Number of channels in the input image
         squeeze_channels (int): Number of squeeze channels
         activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Def
         scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default
     """
 
     def __init__(
         self,
         input_channels: int,
         squeeze_channels: int,
         activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
         scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
     ) -> None:
         super().__init__()
         _log_api_usage_once(self)
         self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
         self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
         self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
         self.activation = activation()
         self.scale_activation = scale_activation()
 
     def _scale(self, input: Tensor) -> Tensor:
         scale = self.avgpool(input)
         scale = self.fc1(scale)
         scale = self.activation(scale)
         scale = self.fc2(scale)
         return self.scale_activation(scale)
 
     def forward(self, input: Tensor) -> Tensor:
         scale = self._scale(input)
         return scale * input

squeeze_channels = max(1, cnf.input_channels // 4)
layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

# project
layers.append(
    Conv2dNormActivation(
        expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
    )
)

# stochastic depth if residual only by defn
from torchvision.ops import StochasticDepth
self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
self.out_channels = cnf.out_channels


FusedMBDeconv

# fused expand: kernel_size != 1
# project
# stochastic depth if residual only by defn


