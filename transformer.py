import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
from vqgan import VQGAN

"""

Here's a summary of how transformers are used in VQ-GAN:

Perturb the quantized latent codes: After obtaining the quantized latent codes z_q 
from the codebook, some of the values in z_q are randomly changed (perturbed) to
create a new set of perturbed latent codes.

Pass the perturbed latent codes through the transformer: The perturbed latent codes 
are then passed through a transformer model, which is pretrained on a large dataset 
(e.g., ImageNet). The transformer tries to predict which latent codes were perturbed.

Calculate the regularization loss: The loss from the transformer prediction is used 
as a regularization term in the overall training loss for the VQ-GAN. The idea behind
this regularization is to encourage the VQ-GAN to generate latent codes that are 
more coherent and meaningful in the context of the pretrained transformer model.

Summary: In summary, the VQGANTransformer class leverages a pretrained VQGAN model
and a GPT-based transformer model to encode images into codebook indices, perturb 
these indices, and use the transformer model to regularize the VQGAN training process
by predicting which indices were perturbed. The class also provides functionality for
sampling new images from the transformer model.

Returns:
    _type_: _description_
"""

class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()

        self.sos_token = args.sos_token

        self.vqgan = self.load_vqgan(args)

        transformer_config = {
            "vocab_size": args.num_codebook_vectors,
            "block_size": 512,
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024
        }
        self.transformer = GPT(**transformer_config)

        self.pkeep = args.pkeep

    """_summary_

    Load VQGAN: A utility function is defined to load a pretrained VQGAN model from a checkpoint.
    
    Returns:
        _type_: _description_
    """
    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        model.load_checkpoint(args.checkpoint_path)
        model = model.eval()
        return model

    """_summary_
    
    Encode to z: This function takes an input image x, encodes it 
    using the VQGAN model, and returns the quantized latent code (quant_z)
    and its corresponding indices in the codebook.
    
    Returns:
        _type_: _description_
    """
    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    """_summary_
    
    z_to_image: This function takes the codebook indices and converts them back to an image using the VQGAN model.

    Returns:
        _type_: _description_
    """
    @torch.no_grad()
    def z_to_image(self, indices, p1=16, p2=16):
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image

    """
    
    Forward: The main forward function takes an input image x, encodes it to 
    codebook indices, and then creates a new set of perturbed indices by randomly
    changing some of the original indices. The perturbed indices are then fed into
    the transformer model to predict which indices were perturbed. The function 
    returns the logits and the target (original) indices.
    
    """
    def forward(self, x):
        _, indices = self.encode_to_z(x)

        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        logits, _ = self.transformer(new_indices[:, :-1])

        return logits, target

    """_summary_
    
    top_k_logits: This function takes the logits and retains only the top k logits
    for each position while setting the rest to -inf. It is used during the sampling process.
    
    """
    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    """
    Sample: This function generates new indices by sampling from the 
    transformer model. It takes an initial set of indices (x) and a 
    context (c), as well as other sampling parameters like the number
    of steps, temperature, and top_k. The generated indices can be used
    to create new images using the z_to_image function.
    
    """
    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for k in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        self.transformer.train()
        return x
    
    """
    
    log_images: This function takes an input image x, encodes it to indices, and then 
    generates a set of images using different sampling strategies: (1) reconstructing 
    the original image, (2) sampling half of the indices, and (3) sampling all of the
    indices. The function returns a dictionary containing these generated images and a
    concatenated tensor of all images for visualization.

    Returns:
        _type_: _description_
    """
    @torch.no_grad()
    def log_images(self, x):
        log = dict()

        _, indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        start_indices = indices[:, :indices.shape[1] // 2]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1])
        half_sample = self.z_to_image(sample_indices)

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, torch.concat((x, x_rec, half_sample, full_sample))
















