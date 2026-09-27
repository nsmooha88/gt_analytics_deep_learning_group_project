# TODO: Unet segmentation model
from torchvision.models import efficientnet_b4
import torch.nn as nn

# the version of efficientnet using imagenet weights as in the paper
# https://www.nature.com/articles/s41598-022-12743-y
# five encoding layers with efficientnet as the encoder
efficientnet_b4(weights='IMAGENET1K_V1')

# this is interesting, might be able to try this pretrained unet?
# https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/

class UNetEncoder(nn.Module):
    '''
    Module for the encoder portion of U-net
    '''

    def __init__(self):
        pass

    def forward(self, input):
        # TODO: convolution structure here
        pass


class UNetDecoder(nn.Module):
    '''
    Module for the decoder portion of U-net
    '''

    def __init__(self):
        pass

    def forward(self, input):
        # TODO: transpose convolution structure here
        pass

class UNet(nn.Module):
    def __init__(self):
        # create unet module here
        # encoder - down: Separate block?
        # conv layers
        # Conv layers

        # decoder - up: Separate block?
        # ConvTranspose layers
        pass

    def forward(self, input):
        pass

