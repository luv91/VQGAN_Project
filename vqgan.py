import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook


class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args).to(device=args.device)
        self.decoder = Decoder(args).to(device=args.device)
        self.codebook = Codebook(args).to(device=args.device)       
        
        # original vq-gan has prequantization and post quantization convolution layers.
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)


    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        
        # pre quantisation layer. 
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        
        
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        
        # post quantization layer.
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)
        
        return decoded_images, codebook_indices, q_loss
    

    # these functions will be used by the transformers separately.. 
    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    # variable lambda is the weighting factor, between vq-vae loss (or perceptual loss)
    # and the GAN loss..  equation 7 in the vq-gan paper
    # G stands for decoder and L stands for the last layer of decoder. 
    def calculate_lambda(self, perceptual_loss, gan_loss):
        
        # last layer and its weight
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        
        # gradients for both percetpaul and GAN loss. 
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        # clipping lambda value between 0 and 10K and then returning lambda. 
        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ
    
    """
    
    The idea behind starting the discriminator later in the training process is to 
    give the generator (in this case, the VQGAN's encoder-decoder architecture) some 
    time to learn to reconstruct images before introducing the adversarial training 
    aspect. This strategy can help stabilize the training process and improve convergence.

    In the initial phase of training, the generator learns to reconstruct images 
    without the pressure of fooling the discriminator. When the discriminator is 
    introduced later, the generator has already learned a decent representation of
    the data, which helps it better cope with the adversarial training aspect. This can
    lead to better performance and stability during training.
    
    """
    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value  # disc_factor is the discriminator weight.. 
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))








