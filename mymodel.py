#Model code taken from the original paper's repository 
#https://github.com/ermongroup/ncsn

import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
from model import *
    
class MyCondRefineBlock(nn.Module):
    def __init__(self, in_planes, features, num_classes, normalizer, act=nn.ReLU(), start=False, end=False):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = 1
        self.features = features
        self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, num_classes, normalizer, act)
        self.rcu_layer = CondRCUBlock(in_planes[0], 2, 2, num_classes, normalizer, act)
        self.conv2d = nn.Conv2d(in_channels=sum([in_planes[i] for i in range(len(in_planes))]), out_channels=self.features, kernel_size=3, stride=1, padding=1)
        self.instance_norm1 = normalizer(in_planes[0], num_classes, bias=True)
        if len(in_planes) == 2:
            self.instance_norm2 = normalizer(in_planes[1], num_classes, bias=True)
        
    def forward(self, xs, y, output_shape):
        if len(xs) == 1:
            # h = self.rcu_layer(xs[0], y)
            h = self.instance_norm1(xs[0], y)
        else:
            # h2 = self.rcu_layer(xs[0], y)
            h2 = self.instance_norm1(xs[0], y)
            h2 = F.interpolate(h2, size=output_shape, mode='bilinear', align_corners=True)
            # h1 = self.rcu_layer(xs[1], y)
            h1 = self.instance_norm2(xs[1], y)
            h = torch.cat([h1, h2], dim=1)
        h = self.conv2d(h)
        h = self.output_convs(h, y)
        return h


class MyCondRefineNetDilated(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        # self.norm = ConditionalInstanceNorm2d
        self.norm = ConditionalInstanceNorm2dPlus
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        self.act = act = nn.ELU()
        # self.act = act = nn.ReLU(True)

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(ngf, self.num_classes)

        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm),
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, dilation=2),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, dilation=2)]
        )

        if config.data.image_size == 28:
            self.res4 = nn.ModuleList([
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                         normalization=self.norm, adjust_padding=True, dilation=4),
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                         normalization=self.norm, dilation=4)]
            )
        else:
            self.res4 = nn.ModuleList([
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                         normalization=self.norm, adjust_padding=False, dilation=4),
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                         normalization=self.norm, dilation=4)]
            )

        self.refine1 = MyCondRefineBlock([2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act, start=True)
        self.refine2 = MyCondRefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act)
        self.refine3 = MyCondRefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, self.num_classes, self.norm, act=act)
        self.refine4 = MyCondRefineBlock([self.ngf, self.ngf], self.ngf, self.num_classes, self.norm, act=act, end=True)

    def _compute_cond_module(self, module, x, y):
        for m in module:
            x = m(x, y)
        return x

    def forward(self, x, y):
        if not self.logit_transform:
            x = 2 * x - 1.
        output = self.begin_conv(x)
        layer1 = self._compute_cond_module(self.res1, output, y)
        layer2 = self._compute_cond_module(self.res2, layer1, y)
        layer3 = self._compute_cond_module(self.res3, layer2, y)
        layer4 = self._compute_cond_module(self.res4, layer3, y)
        
        ref1 = self.refine1([layer4], y, layer4.shape[2:])
        ref2 = self.refine2([ref1, layer3], y, layer3.shape[2:])
        ref3 = self.refine3([ref2, layer2], y, layer2.shape[2:])
        output = self.refine4([ref3, layer1], y, layer1.shape[2:])
        
        output = self.normalizer(output, y)
        output = self.act(output)
        output = self.end_conv(output)
        
        return output


