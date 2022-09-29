import torch, numpy as np
import torchvision
from Dataset import AbstractPortraitDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from Discriminator import Discriminator
from Generator import Generator
from utils import weights_init_normal

def train(disc_A, disc_P, gen_A, gen_P, dataloader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    A_reals = 0
    A_fakes = 0

    loop = tqdm(dataloader, leave=True) # progress bar

    for idx, (abstract, portrait) in enumerate(loop):
        abstract = abstract.to(config.DEVICE)
        portrait = portrait.to(config.DEVICE)

        # Train the Discriminators A and P
        with torch.cuda.amp.autocast():
            opt_disc.zero_grad()

            fake_abstract = gen_A(portrait)
            D_A_real = disc_A(abstract)
            D_A_fake = disc_A(fake_abstract.detach())
            A_reals += D_A_real.mean().item()
            A_fakes += D_A_fake.mean().item()
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            fake_portrait = gen_P(abstract)
            D_P_real = disc_P(portrait)
            D_P_fake = disc_P(fake_portrait.detach())
            D_P_real_loss = mse(D_P_real, torch.ones_like(D_P_real))
            D_P_fake_loss = mse(D_P_fake, torch.zeros_like(D_P_fake))
            D_P_loss = D_P_real_loss + D_P_fake_loss

            # put together
            D_loss = (D_A_loss + D_P_loss) / 2
            opt_disc.step()

        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators A and P
        with torch.cuda.amp.autocast():
            opt_gen.zero_grad()
            # adversarial loss for both generators
            D_A_fake = disc_A(fake_abstract)
            D_P_fake = disc_P(fake_portrait)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_P = mse(D_P_fake, torch.ones_like(D_P_fake))

            # cycle loss
            cycle_abstract = gen_A(fake_portrait)
            cycle_portrait = gen_P(fake_abstract)
            cycle_abstract_loss = l1(abstract, cycle_abstract)
            cycle_portrait_loss = l1(portrait, cycle_portrait)

            identity_abstract = gen_A(abstract)
            identity_portrait = gen_P(portrait)
            identity_abstract_loss = l1(abstract, identity_abstract)
            identity_portrait_loss = l1(portrait, identity_portrait)

            # add all together
            G_loss = (
                    loss_G_A
                    + loss_G_P
                    + cycle_abstract_loss * config.LAMBDA_CYCLE
                    + cycle_portrait_loss * config.LAMBDA_CYCLE
                    + identity_abstract_loss * config.LAMBDA_IDENTITY  # oder
                    + identity_portrait_loss * config.LAMBDA_IDENTITY  # andersum?
            )

            opt_gen.step()

        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            # *0.5+0.5 = Inverse of the normalization to get the correct coloring
            save_image(fake_abstract*0.5+0.5, f"saved_images/abstract_{idx}.png")

        loop.set_postfix(A_real=A_reals / (idx + 1), A_fake=A_fakes / (idx + 1))

def main():
    disc_A = Discriminator(3).to(config.DEVICE)  # classify images of Abstract images
    disc_P = Discriminator(3).to(config.DEVICE)  # discriminates if its a real portrait or a fake (abstract) portrait
    gen_A = Generator(3, 9).to(config.DEVICE)  # generates an abstract image
    gen_P = Generator(3, 9).to(config.DEVICE)  # takes in an image (abstract) and generates an portrait image

    disc_A.apply(weights_init_normal)
    disc_P.apply(weights_init_normal)
    gen_A.apply(weights_init_normal)
    gen_P.apply(weights_init_normal)

    """ Optimizers """
    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_P.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),  # 0.5 for the momentum turn and 0.999 for beta2
    )

    opt_gen = optim.Adam(
        list(gen_P.parameters()) + list(gen_A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    l1 = nn.L1Loss()  # cycle consistency loss and identity loss
    mse = nn.MSELoss()  # mean squared error

    train_dataset = AbstractPortraitDataset(
        root_abstract="datasets/train/A",
        root_portrait="datasets/train/P", transform=config.transforms
    )

    val_dataset = AbstractPortraitDataset(
        root_abstract="datasets/test/A",
        root_portrait="datasets/test/P", transform=config.transforms
    )

    loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # float 16 training, ohne in float 32
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train(disc_A, disc_P, gen_P, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler)

        """ Save models checkpoints"""
        torch.save(disc_A.state_dict(), 'network-output/disc_A.pth')
        torch.save(disc_P.state_dict(), 'network-output/disc_P.pth')
        torch.save(gen_A.state_dict(), 'network-output/gen_A.pth')
        torch.save(gen_P.state_dict(), 'network-output/gen_P.pth')

if __name__ == "__main__":
    main()