import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, detach=False):
    if backbone == 'resnet50':
        return resnet.ResNet50(output_stride, BatchNorm, detach=detach)
    elif backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm, detach=detach)
#    elif backbone == 'xception':
#        return xception.AlignedXception(output_stride, BatchNorm)
#    elif backbone == 'drn':
#        return drn.drn_d_54(BatchNorm)
#    elif backbone == 'mobilenet':
#        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
