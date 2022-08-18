import torch
import torch.nn as nn
import torch.nn.functional as F

""" PatchGAN Discriminator """


class Discriminator(nn.Module):
    def __init__(self, n_input):
        super(Discriminator, self).__init__()

        """ Convolution layers one after another """

        layer = [nn.Conv2d(n_input, 64, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        layer += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        layer += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        layer += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        layer += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*layer)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0])
