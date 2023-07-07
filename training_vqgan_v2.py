# python training_vqgan_v2.py --checkpoint-path "C:/Users/luvve/VQGAN/checkpoints/vqgan_epoch_7.pt"
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


class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)
        self.resumed_epoch = 0
        if args.checkpoint_path is not None:
            self.vqgan.load_state_dict(torch.load(args.checkpoint_path))
            resumed_epoch = int(args.checkpoint_path.split('_')[-1].split('.')[0])
            self.resumed_epoch = resumed_epoch
            print(f"Resuming training from checkpoint: {args.checkpoint_path}")
            print(f"Resuming from epoch {resumed_epoch}")

        if args.checkpoint_vq_opt_path is not None and args.checkpoint_disc_opt_path is not None:
            self.opt_vq.load_state_dict(torch.load(args.checkpoint_vq_opt_path))
            self.opt_disc.load_state_dict(torch.load(args.checkpoint_disc_opt_path))
            
        self.prepare_training()

        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        # vq_vae part optimizer
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        # discriminator part. 
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        train_dataset = load_data(args)
        steps_per_epoch = len(train_dataset)
        for epoch in range(self.resumed_epoch+1,args.epochs):
            # The tqdm function is being used here to provide a progress bar for each epoch of training. 
            with tqdm(range(len(train_dataset))) as pbar:
                
                # In this particular case, the zip function is used to iterate 
                # over both the progress bar and the training dataset simultaneously.
                for i, imgs in zip(pbar, train_dataset):
                    # imgs.shape torch.Size([5, 3, 256, 256]) ==> first dimension is batch size(5)
                    # remaining dimension is 3,256,256
                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                    # Backward loss.. 
                    
                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    if i % 10 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs[:4], decoded_images.add(1).mul(0.5)[:4]))
                            vutils.save_image(real_fake_images, os.path.join("results", f"{epoch}_{i}.jpg"), nrow=4)

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(0)
                torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt"))
                
                # Saving the optimizer state
                torch.save(self.opt_vq.state_dict(), os.path.join("checkpoints", f"opt_vq_epoch_{epoch}.pt"))
                torch.save(self.opt_disc.state_dict(), os.path.join("checkpoints", f"opt_disc_epoch_{epoch}.pt"))
                


                


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=5, help='Input batch size for training (default: 6)')
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
    
    args.dataset_path = r"C:\Users\luvve\flowersSix\jpg"
    
    
    # args.dataset_path = r"C:\Users\luvve\flowers\jpg"
    # args.checkpoint_path = r"C:\Users\luvve\VQGAN\checkpoints\vqgan_epoch_61.pt"
    # args.checkpoint_vq_opt_path = r"C:\Users\luvve\VQGAN\checkpoints\opt_vq_epoch_61.pt"
    # args.checkpoint_disc_opt_path = r"C:\Users\luvve\VQGAN\checkpoints\opt_disc_epoch_61.pt"

    train_vqgan = TrainVQGAN(args)



