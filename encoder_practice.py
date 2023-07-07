import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, weights_init
from torchinfo import summary

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
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=100, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Path to the checkpoint to resume training from (default: None)')
    parser.add_argument('--checkpoint-vq-opt-path', type=str, default=None, help='Path to the VQ optimizer checkpoint to resume training from (default: None)')
    parser.add_argument('--checkpoint-disc-opt-path', type=str, default=None, help='Path to the discriminator optimizer checkpoint to resume training from (default: None)')

    args = parser.parse_args()
    # Create random input tensor
    # batch_size = 32
    # image_channels = 3
    # height = width = 256
    input_tensor = torch.randn(args.batch_size, args.image_channels, args.image_size, args.image_size)

    # # Create an instance of Encoder
    # latent_dim = 256  # define this as per your requirement
    encoder = Encoder(args)

    # # Print an input and output summary of our Transformer Encoder (uncomment for full output)
    random_input_image = (1, 3, 256, 256)
    summary(model=encoder,
            input_size = random_input_image, 
            col_names=["input_size", "output_size", "num_params", "trainable"])
    
    # Pass the input tensor through the encoder
    # output = encoder(input_tensor)
        