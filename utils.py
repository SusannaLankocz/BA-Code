import random, torch, os, numpy as np
import torch.nn as nn
import config
import copy
from torch.autograd import Variable


""" To initialize weights of a network. """
def weights_init_normal(m):
   classname = m.__class__.__name__
   if classname.find('Conv2d') != -1:
       torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
   elif classname.find('BatchNorm2d') != -1:
       torch.nn.init.normal(m.weight.data, 1.0, 0.02)
       torch.nn.init.constant(m.bias.data, 0.0)