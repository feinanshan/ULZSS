import json
from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.context460_loader import context460Loader
from ptsemseg.loader.context60_loader import context60Loader
from ptsemseg.loader.ade20k_loader import ade20kLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "pascalcontext460": context460Loader,
        "pascalcontext60": context60Loader,
        "ade20k": ade20kLoader,
    }[name]
