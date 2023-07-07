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

import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish

# decoder, opposite of encoder, however little bigger than it. 
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        attn_resolutions = [16]
        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        # first 4 layers of the decoder. 
        layers = [nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

        # all other layers. 
        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, 3, 1, 1))
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
    decoder = Decoder(args)

    # # Print an input and output summary of our Transformer Encoder (uncomment for full output)
    random_input_image = (1, 256, 16, 16)
    summary(model=decoder,
            input_size = random_input_image, 
            col_names=["input_size", "output_size", "num_params", "trainable"])
    
    # Pass the input tensor through the encoder
    # output = encoder(input_tensor)
        