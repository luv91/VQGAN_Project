import torch
import torch.nn as nn

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
        term1 = torch.mean((z_q.detach() - z)**2)
        
        term2 = torch.mean((z_q - z.detach())**2)

        loss =  term1+ self.beta * term2
        # To bypass the non-differentiable operation of selecting the closest codebook vector, 
        # a straight-through estimator is employed. 
        z_q = z + (z_q - z).detach()  # preserving the gradients for the backward flow. 

        z_q = z_q.permute(0, 3, 1, 2)

        # returning quantized latent vectors, indices of these codebook vectors, and loss
        # indices will be used later in the transformer part. 
        return z_q, min_encoding_indices, loss