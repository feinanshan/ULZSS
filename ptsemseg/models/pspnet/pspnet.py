import torch,pdb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import resnet
from ptsemseg.loss import OhemCELoss2D,CrossEntropyLoss

from torch.autograd import Variable

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class PSPNet(nn.Module):

    """
    Pyramid Scene Parsing Network
    URL: https://arxiv.org/abs/1612.01105

    References:
    1) Original Author's code: https://github.com/hszhao/PSPNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/PSPNet-tensorflow

    Visualization:
    http://dgschwend.github.io/netscope/#/gist/6bfb59e6a3cfcb4e2bb8d47f827c2928

    """

    def __init__(self,
            nclass=21,
            se_loss=None,
            norm_layer=nn.BatchNorm2d,
            backbone='resnet101',
            root='~/.encoding/models',
            dilated=True,
            aux=True,
            aux_weight=0.4,
            multi_grid=True,
            loss_fn=None
        ):
        super(PSPNet, self).__init__()
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.loss_fn = loss_fn
        self.nclass = nclass
        self.aux = aux
        self.aux_weight = aux_weight
        # copying modules from pretrained models
        self.backbone = backbone


        if backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=True, dilated=dilated, multi_grid=multi_grid,
                                               deep_base=False, norm_layer=norm_layer)
            self.expansion = 1
        elif backbone == 'resnet34':
            self.pretrained = resnet.resnet34(pretrained=True, dilated=dilated, multi_grid=multi_grid,
                                               deep_base=False, norm_layer=norm_layer)
            self.expansion = 1
        elif backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True, dilated=dilated,multi_grid=multi_grid,
                                              norm_layer=norm_layer, root=root)
            self.expansion = 4
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=True, dilated=dilated,multi_grid=multi_grid,
                                               norm_layer=norm_layer, root=root)
            self.expansion = 4
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=True, dilated=dilated,multi_grid=multi_grid,
                                               norm_layer=norm_layer, root=root)
            self.expansion = 4
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        
        # bilinear upsample options
        
        self.head = PSPHead(512*self.expansion, nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(256*self.expansion, nclass, norm_layer)

        #sss=0
        #for parameters in self.parameters():
        #    sss= sss+1
        #print(sss)

        #print(len(self.get_params()[0])+len(self.get_params()[1])+len(self.get_params()[2])+len(self.get_params()[3]))

        #pdb.set_trace()

    def forward(self, x, lbl=None):
        _, _, h, w = x.size()

        #pdb.set_trace()
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        x = self.head(c4)
        outputs = F.interpolate(x, (h,w), **self._up_kwargs)
        # Auxiliary layers for training
        if self.training:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h,w), **self._up_kwargs)

            loss = self.loss_fn(outputs,lbl)+self.aux_weight*self.loss_fn(auxout,lbl)
            return loss
        else:
            return outputs

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():

            if isinstance(child,(OhemCELoss2D,CrossEntropyLoss)):
                continue
            elif isinstance(child, (PSPHead, PyramidPooling,FCNHead)):
                child_wd_params, child_nowd_params = child.get_params()
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                child_wd_params, child_nowd_params = child.get_params()
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params





class PSPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4

        self.norm_layer = norm_layer
        self.conv5 = nn.Sequential(PyramidPooling(in_channels, norm_layer, up_kwargs),
                                   nn.Conv2d(in_channels * 2, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))
        self.init_weight()

    def forward(self, x):
        return self.conv5(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Sequential):
                for lz in ly:
                    if isinstance(lz, nn.Conv2d):
                        nn.init.kaiming_normal_(lz.weight, a=1)
                        if not lz.bias is None: nn.init.constant_(lz.bias, 0)
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.norm_layer)):
                nowd_params += list(module.parameters())

        return wd_params, nowd_params



class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.norm_layer = norm_layer
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs
        self.init_weight()


    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Sequential):
                for lz in ly:
                    if isinstance(lz, nn.Conv2d):
                        nn.init.kaiming_normal_(lz.weight, a=1)
                        if not lz.bias is None: nn.init.constant_(lz.bias, 0)
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


    def get_params(self):
        wd_params, nowd_params = [], []

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.norm_layer)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs={}, with_global=False):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self._up_kwargs = up_kwargs
        self.norm_layer = norm_layer
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))
        self.init_weight()

    def forward(self, x):
        return self.conv5(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Sequential):
                for lz in ly:
                    if isinstance(lz, nn.Conv2d):
                        nn.init.kaiming_normal_(lz.weight, a=1)
                        if not lz.bias is None: nn.init.constant_(lz.bias, 0)
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.norm_layer)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
