import torch
import torch.nn as nn


""" Resnet Generator """

class ResNetBlock(nn.Module):
    def __init__(self, in_features):
        super(ResNetBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    """ forward function with skip connections """
    def forward(self, x):
        return x + self.conv_block(x)



class Generator(nn.Module):
    def __init__(self, n_input, n_output, n_resnet_blocks=9):
        super(Generator, self).__init__()

        """ Initial conv block"""
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(n_input, 64, kernel_size=7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(True)]

        """ Add Downsampling layers"""
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(True)]
            in_features = out_features
            out_features = in_features * 2

        """ Add ResNet blocks"""
        for _ in range(n_resnet_blocks):
            model += [ResNetBlock(in_features)]

        """ Add Upsampling layers"""
        out_features = in_features//2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(True)]
            in_features = out_features
            out_features = in_features//2

            """ Output Layer"""
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(64, n_output, kernel_size=7, padding=0)]
            model += [nn.Tanh()]

            self.model = nn.Sequential(*model)

        def forward(self, input):
            """Standard forward """
            return self.model(input)