import copy
import torchvision.models as models
import torch.nn as nn

#from encoding.nn import SyncBatchNorm

from ptsemseg.models.pspnet import PSPNet
from ptsemseg.models.deeplabv3p import DeepLabV3P

class BatchNorm2d(nn.BatchNorm2d):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, num_features, activation='none'):
        super(BatchNorm2d, self).__init__(num_features=num_features)
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'none':
            self.activation = lambda x:x
        else:
            raise Exception("Accepted activation: ['leaky_relu']")

    def forward(self, x):
        return self.activation(super(BatchNorm2d, self).forward(x))





def get_model(model_dict, nclass, loss_fn=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")
    param_dict["loss_fn"] = loss_fn
    if param_dict['syncBN']:
        param_dict['norm_layer'] = SyncBatchNorm
    else:
        param_dict['norm_layer'] = BatchNorm2d

    param_dict.pop('syncBN')
    
    model = model(nclass=nclass, **param_dict)
    return model


def _get_model_instance(name):
    try:
        return {
            "pspnet": PSPNet,
            "deeplabv3p": DeepLabV3P,
        }[name]
    except:
        raise ("Model {} not available".format(name))
