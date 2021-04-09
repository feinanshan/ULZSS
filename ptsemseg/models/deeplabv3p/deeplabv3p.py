import torch,pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ptsemseg.loss import OhemCELoss2D,CrossEntropyLoss, BCELoss
from backbone import build_backbone
from backbone.resnet import ResNet
import cv2
import pdb

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class DeepLabV3P(nn.Module):

    def __init__(self,
                 nclass=21,
                 output_stride=None,
                 backbone='resnet101',
                 norm_layer=None,
                 loss_fn=None,
                 detach_backbone=True):
        super(DeepLabV3P, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        self.loss_fn = loss_fn

        self._up_kwargs = up_kwargs
        self.nclass = nclass

        self.backbone = build_backbone(backbone, output_stride, norm_layer, detach=detach_backbone)
        self.aspp = ASPP(backbone, output_stride, norm_layer)
        self.decoder = Decoder(self.nclass, backbone, norm_layer)
        self.varmaping = VarMapping(300, 128,BatchNorm=norm_layer)

        #sss=0
        #for parameters in self.parameters():
        #    sss= sss+1
        #print(sss)

        #print(len(self.get_params()[0])+len(self.get_params()[1])+len(self.get_params()[2])+len(self.get_params()[3]))

        #pdb.set_trace()

    def forward(self, images, labels, embds, ignr_idx=250):
        features = self.forward_get_fea(images)
        output_all = self.output_pred(features, labels, embds, ignr_idx)

        if self.training:
            return sum(output_all)
        else:
            return  output_all

    def forward_get_fea(self, input): # sem_emb [C,dim]
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        fea = self.decoder.forward_before_class_prediction(x, low_level_feat)
        return fea

    def output_pred(self, features, labels, embds, ignr_idx=250):
        output_all = []
        for (fea_i, lbl_i) in zip(features, labels):
            output = self.output_inst_pred(fea_i, lbl_i, embds, ignr_idx)
            output_all = output_all + output
        return  output_all

    def output_inst_pred(self, fea_i, lbl_i, embds, ignr_idx=250):
        c, h, w = fea_i.size()

        features_i = (
            fea_i.permute(1, 2, 0)
                .contiguous()
                .view((-1, c))
        )

        target_i = nn.functional.interpolate(
            lbl_i.view(1, 1, lbl_i.shape[0], lbl_i.shape[1]),
            size=(h, w),
            mode="nearest",
        ).view(-1).long()

        fea_con = self.decoder.map_cond(features_i).mean(0).view(1,-1)

        fea_con = fea_con.expand(h*w,-1) 

        unique_class = torch.unique(target_i)

        output = []
        for idx_in in unique_class:
            if idx_in.cpu().item() is not ignr_idx:
                idx_mask = (target_i == idx_in).view(1, 1, h, w).contiguous().float()

                fill_max = torch.full(target_i.size(), idx_in).long().cuda()
                sem_map = embds(fill_max).data

                # global 
                gl_map = torch.cat([sem_map, fea_con], dim=1)
                g_delta = self.decoder.global_delta(gl_map)
                g_delta = g_delta.view(1, h, w, -1).contiguous().permute(0, 3, 1, 2).mean()
                

                proto = self.varmaping(sem_map)
                fea_cat = torch.cat([features_i, proto], dim=1)
                fea_cat = fea_cat.view(1, h, w, -1).contiguous().permute(0, 3, 1, 2)

                pred_fea = self.decoder.pred_layer(fea_cat)

                l_mu_ = self.decoder.pred_mu(pred_fea)
                l_mu = self.decoder.sigmoid(l_mu_)

                s_delta = self.decoder.pred_delta(pred_fea.detach())
                l_delta = s_delta.exp()
                l_delta = torch.clamp(l_delta, min=0.0001,max=10000)
                g_delta = g_delta.exp()


                if self.training:
                    loss_g = self.dice_loss(l_mu, idx_mask)/(g_delta)+torch.log(g_delta.sqrt())
                    loss_l = (self.loss_fn(l_mu_, idx_mask)/(l_delta)).mean()+ torch.log(l_delta.sqrt()).mean()
                    loss_ = loss_g + 0.01*loss_l
                    #loss_ = 
                    output.append(loss_)
                else:
                    M,I,U = self.compute_iou_for_binary_segmentation(l_mu,idx_mask)
                    output.append([int(idx_in.cpu().data),M,I,U])
                    #pdb.set_trace()

        return output

    def dice_loss(self, inputs, target, alpha=0):
        self.smooth = 1 
        self.p = 2

        inputs = inputs.contiguous().view(inputs.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(inputs, target), dim=1) + self.smooth
        den = torch.sum(inputs.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - 2*num / den

        return loss

    def compute_iou_for_binary_segmentation(self, pred_mask, target):
        pred_mask = np.array((pred_mask.cpu().data[0,0])>=0.5)
        target = np.array((target.cpu().data[0,0])>=0.5)
        M = (pred_mask==1).sum()+0.0000001
        I = np.logical_and(pred_mask == 1, target == 1).sum()
        U = np.logical_or(pred_mask == 1, target == 1).sum()+0.0000001

        return M,I,U

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if isinstance(child,(OhemCELoss2D,CrossEntropyLoss,BCELoss,ResNet)):
                continue
            elif isinstance(child, (ASPP,Decoder,VarMapping)):
                child_wd_params, child_nowd_params = child.get_params()
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:

                child_wd_params, child_nowd_params = child.get_params()
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.BatchNorm = BatchNorm
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes, activation='leaky_relu')

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, self.BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_params(self):
        wd_params, nowd_params = [], []

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.BatchNorm)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        self.BatchNorm = BatchNorm
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256, activation='leaky_relu')
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)

        return self.dropout(x)

    def get_params(self):
        wd_params, nowd_params = [], []

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.BatchNorm)):
                nowd_params += list(module.parameters())

        return wd_params, nowd_params


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, self.BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        self.BatchNorm = BatchNorm

        if backbone == 'resnet101' or backbone == 'resnet50' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48, activation='leaky_relu')
   
        self.map_cond = nn.Sequential(nn.Linear(256, 64),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(64, 64),
                                        nn.ReLU())


        self.global_delta = nn.Sequential(nn.Linear(364, 64),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(64, 1))


        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))

        self.pred_layer = nn.Sequential(nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0),
                                        nn.ReLU(),
                                        nn.Dropout(0.5))

        self.pred_mu = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False))

        self.sigmoid = nn.Sigmoid()

        self.pred_delta = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False))
        self._init_weight()


    def forward_before_class_prediction(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x


    def forward(self, x, low_level_feat):
        x = self.forward_before_class_prediction(x, low_level_feat)
        x = self.pred_layer(x)
        return x


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, self.BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()               


    def get_params(self):
        wd_params, nowd_params = [], []

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.BatchNorm)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class VarMapping(nn.Module):
    def __init__(self, embed_dim=300, out_dim=128, BatchNorm=None):
        super(VarMapping, self).__init__()

        self.BatchNorm = BatchNorm
        self.mean = nn.Linear(embed_dim, out_dim)
        self.delta = nn.Linear(embed_dim, out_dim)
        self._init_weight()

    def forward(self, embd):
        mean = self.mean(embd)

        return mean

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, self.BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_params(self):
        wd_params, nowd_params = [], []

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.BatchNorm)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
