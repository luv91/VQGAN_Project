import torch
import torch.nn as nn
from torchinfo import summary
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
"""

 The input to the Codebook is the output of the encoder. 
 In the context of VQ-GANs, the encoder processes the input image and generates 
 a continuous latent representation z with shape [batch_size, latent_dim, height, width]. 
 This continuous latent representation is then fed into the Codebook, where it is 
 discretized into a quantized tensor z_q using the learned codebook vectors.
 
 from: https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/ (Nice figure: Figure: Vector quantization 
 is a classic signal processing technique that finds
 the representative centroids for each cluster.)
 
 
 The codebook is generated through a process called vector quantization (VQ), 
 i.e., the “VQ” part of “VQGAN.” Vector quantization is a signal processing technique for 
 encoding vectors. It represents all visual parts found in the convolutional step in a
 quantized form, making it less computationally expensive once passed to a transformer network.


One can think of vector quantization as a process of dividing vectors into groups 
that have approximately the same number of points closest to them (Ballard, 1999). 
Each group is then represented by a centroid (codeword), usually obtained via k-means or 
any other clustering algorithm. In the end, one learns a dictionary of centroids (codebook)
and their corresponding members.


Returns:
    _type_: _description_
"""
class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta  # weighting factor for the codebook loss.. 
        # self.embedding is the embedding matrix
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        # initialize weights by uniform distribution. 
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        
        # The input tensor z has a shape of [batch_size, latent_dim, height, width].
        # It is permuted to a shape of [batch_size, height, width, latent_dim]
        z = z.permute(0, 2, 3, 1).contiguous()        
        z_flattened = z.view(-1, self.latent_dim)
        
        #z_flattened = z_flattened.to(self.embedding.weight.device)
        #self.embedding = self.embedding.to(self.embedding.weight.device)
    
        # expanded version of the l2 loss: (a-b)**2 = a**2-2.a.b+b**2
        # d = (z_flattened-self.embedding.weight)**2
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        # The index of the codebook vector with the smallest distance to each element in
        # z_flattened is found using torch.argmin(d, dim=1) and stored in min_encoding_indices.
        # find the closest vectors. 
        min_encoding_indices = torch.argmin(d, dim=1)
        
        # actual codebook vectors from indices matrix.. 
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # codebook loss.. stopgradient part (stopgradient part is done by detach function in torch. )
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        # To bypass the non-differentiable operation of selecting the closest codebook vector, 
        # a straight-through estimator is employed. 
        z_q = z + (z_q - z).detach()  # preserving the gradients for the backward flow. 

        z_q = z_q.permute(0, 3, 1, 2)

        # returning quantized latent vectors, indices of these codebook vectors, and loss
        # indices will be used later in the transformer part. 
        return z_q, min_encoding_indices, loss
    
    
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
    
    #input_tensor = torch.randn(args.batch_size, args.image_channels, args.image_size, args.image_size)
    #input_tensor = input_tensor.to(args.device)
    
    # # Create an instance of Encoder
    # latent_dim = 256  # define this as per your requirement
    codebook = Codebook(args).to(args.device)

    # # Print an input and output summary of our Transformer Encoder (uncomment for full output)
    random_input_image = torch.randn(256,256,1).to(args.device)
    
    summary(model=codebook,
            input_size = random_input_image, 
            col_names=["input_size", "output_size", "num_params", "trainable"])
    
    # Pass the input tensor through the encoder
    output = codebook(random_input_image)
    print("output.shape", output.shape)
        