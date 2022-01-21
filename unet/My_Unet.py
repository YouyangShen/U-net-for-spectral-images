# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 22:03:26 2021

@author: Youyang Shen
"""

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 28)
        self.down1 = Down(28, 56)
        self.down2 = Down(56, 112)
        self.down3 = Down(112, 224)
        factor = 2 if bilinear else 1
        self.down4 = Down(224, 448 // factor)
        self.up1 = Up(448, 224 // factor, bilinear)
        self.up2 = Up(224, 112 // factor, bilinear)
        self.up3 = Up(112, 56 // factor, bilinear)
        self.up4 = Up(56, 28, bilinear)
        self.outc = OutConv(28, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
