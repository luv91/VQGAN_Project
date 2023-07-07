import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish

# Encoder block ==> Page 11/52 (pdf): https://arxiv.org/pdf/2012.09841.pdf

"""_summary_

Step-by-step breakdown of the Encoder module:

Module 1 - Initial Convolution Layer

Layer 1: Conv2d layer (input: image_channels, output: channels[0], kernel_size: 3, stride: 1, padding: 1)
Module 2 - Residual Blocks and Non-local Blocks

For each pair of consecutive channels in the channels list:
Layer 2a: ResidualBlock (input: in_channels, output: out_channels)
Layer 2b: ResidualBlock (input: out_channels, output: out_channels)
If the current resolution is in attn_resolutions, add a NonLocalBlock (input: out_channels)
If not at the last pair of consecutive channels, add a DownSampleBlock (input: out_channels)

Module 3 - Final Layers

Layer 3a: ResidualBlock (input: channels[-1], output: channels[-1])
Layer 3b: NonLocalBlock (input: channels[-1])
Layer 3c: ResidualBlock (input: channels[-1], output: channels[-1])
Layer 3d: GroupNorm (input: channels[-1])
Layer 3e: Swish activation
Layer 3f: Conv2d layer (input: channels[-1], output: latent_dim, kernel_size: 3, stride: 1, padding: 1)
The entire architecture is then built using a nn.Sequential layer containing all the layers from Module 1, Module 2, and Module 3.

channels = [128, 128, 128, 256, 256, 512]
There are 5 instances of Module 2 for the given channel list.

Here's the breakdown:

Module 2.1: Transition from channels[0] (128) to channels[1] (128)
Module 2.2: Transition from channels[1] (128) to channels[2] (128)
Module 2.3: Transition from channels[2] (128) to channels[3] (256)
Module 2.4: Transition from channels[3] (256) to channels[4] (256)
Module 2.5: Transition from channels[4] (256) to channels[5] (512)

The NonLocalBlock is added based on the attn_resolutions list. 
In the given code, attn_resolutions = [16]. The purpose of using the NonLocalBlock is 
to model long-range dependencies within the input feature maps by capturing spatial relationships at specific resolutions.



Returns:
_type_: _description_
"""


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        channels = [128, 128, 128, 256, 256, 512]
        
        attn_resolutions = [16]  # downscaling, so decreasing the resolution. 
        
        num_res_blocks = 2
        
        resolution = 256
        
        layers = [nn.Conv2d(args.image_channels, channels[0], 3, 1, 1)]
        
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
                    
            """
            
            The DownSampleBlock is responsible for reducing the spatial dimensions 
            (height and width) of the feature maps, while the ResidualBlock layers handle
            the change in the number of channels. For the last two consecutive channels, the 
            model does not add a DownSampleBlock to maintain the spatial resolution and fully 
            utilize the extracted features before mapping them to the latent space. The change 
            from 256 to 512 channels is managed by the ResidualBlock layers.
            
            """
            if i != len(channels)-2: # If not at the last pair of consecutive channels, add a DownSampleBlock (input: out_channels)
                
                # The purpose of the DownSampleBlock is to reduce the spatial dimensions
                # (height and width) of the feature maps, while the number of channels changes 
                # within the ResidualBlock layers. 
                layers.append(DownSampleBlock(channels[i+1]))
                resolution //= 2
                
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
        """_summary_
        The output of the encoder, will be given to the codebook:
        Check out figure: Figure: VQGAN employs a codebook as an intermediary representation before 
        feeding it to a transformer network. The codebook is then learned using vector quantization (VQ).
        
        https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/
        
        """