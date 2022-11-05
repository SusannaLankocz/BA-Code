import torch
import sys
from Dataset import AbstractPortraitDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from Generator import Generator
import config

# Networks
gen_A = Generator(3, 9).to(config.DEVICE)  # generates an abstract image
gen_P = Generator(3, 9).to(config.DEVICE)  # takes in an image (abstract) and generates an portrait image

gen_A.load_state_dict(torch.load('network-output/gen_A.pth'))
gen_P.load_state_dict(torch.load('network-output/gen_P.pth'))

gen_A.eval()
gen_P.eval()

val_dataset = AbstractPortraitDataset(
        root_abstract="datasets/test/A",
        root_portrait="datasets/test/P", transform=config.transforms
    )

val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

###### Testing ######

for i, (abstract, portrait) in enumerate(val_loader):
    abstract = abstract.to(config.DEVICE)
    portrait = portrait.to(config.DEVICE)

    # Generate output
    fake_abstract = gen_A(portrait)* 0.5 + 0.5
    fake_portrait = gen_P(abstract)* 0.5 + 0.5

    # Save image files
    save_image(fake_abstract, 'output/A/%04d.png' % (i+1))
    save_image(fake_portrait, 'output/B/%04d.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(val_loader)))
