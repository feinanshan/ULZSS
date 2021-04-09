import os
import torch
import numpy as np
import scipy.misc as m
import scipy.io as scio
import cv2
from torch.utils import data
import torchvision.transforms.functional as TF
from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale

def color_map( N=40, normalized=False):
    """
    Return Color Map in PASCAL VOC format
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255.0 if normalized else cmap
    return cmap

class ade20kLoader(data.Dataset):

    colors = color_map(N=256)
    label_colours = dict(zip(range(256), colors))

    def __init__(
        self,
        cfg,
        mode="train",
        augmentations=None,
    ):
        self.transductive = cfg["transductive"]
        self.root = cfg["data_path"]
        self.mode = mode
        self.augmentations = augmentations
        if mode=='train':
            self.split = cfg["train_split"]
        elif mode=='val':
            self.split = cfg["val_split"]

        self.files = {}
        self.images_base = os.path.join(self.root, "IMG", self.split)
        self.annotations_base = os.path.join(self.root, "GT", self.split)
        self.files[self.split] = recursive_glob(rootdir=self.annotations_base, suffix=".png")

        self.n_classes = 150
        self.void_classes = [0]
        self.valid_classes = range(1,151)

        self.unseen75 = [1, 3, 4, 5, 6, 11, 13, 15, 18, 19, 
                        20, 21, 24, 26, 29, 30, 31, 33, 34, 35, 
                        38, 41, 43, 44, 45, 48, 50, 53, 55, 58, 
                        59, 62, 63, 64, 70, 71, 75, 76, 77, 80, 
                        82, 83, 84, 86, 87, 91, 92, 93, 94, 98, 
                        99, 100, 104, 105, 109, 110, 112, 115, 120, 123, 
                        124, 125, 126, 127, 128,129, 133, 136, 137, 138, 
                        139, 140, 143, 144, 147]#Note that these should be choosen from IDs for valid categories

        self.unseen50 = [1, 3, 4, 5, 6, 13, 15, 18, 19, 20, 
                        21, 24, 26, 29, 31, 33, 34, 35, 41, 43, 
                        48, 50, 53, 58, 59, 62, 64, 71, 75, 80, 
                        82, 83, 84, 86, 87, 91, 92, 93, 99, 109, 
                        112, 123, 126, 127, 129, 133, 137, 138, 144, 147]

        self.unseen25 = [1, 3, 6, 13, 15, 18, 21, 24, 29, 34, 
                        35, 41, 50, 53, 71, 82, 83, 84, 91, 92, 
                        99, 109, 126, 127, 133]

        self.unseen5 = [3, 18, 34, 50, 83]

        if cfg["unseen"] ==0:
            self.unseen_classes = [] # Note that these should be choosen from IDs for valid categories
        elif cfg["unseen"] ==5:
            self.unseen_classes = self.unseen5  # Note that these should be choosen from IDs for valid categories
        elif cfg["unseen"]==25:
            self.unseen_classes = self.unseen25  # Note that these should be choosen from IDs for valid categories
        elif cfg["unseen"]==50:
            self.unseen_classes = self.unseen50  # Note that these should be choosen from IDs for valid categories
        elif cfg["unseen"]==75:
            self.unseen_classes = self.unseen75  # Note that these should be choosen from IDs for valid categories
        else:
            raise Exception("Only support 5,25,50,75 unseen")

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(150)))

        self.embd = scio.loadmat(cfg["emdb_path"])["embd"]
        self.embd = self.embd[self.valid_classes]
        self.embeddings = torch.nn.Embedding(self.embd.shape[0], self.embd.shape[1])
        self.embeddings.weight.requires_grad = False
        self.embeddings.weight.data.copy_(torch.from_numpy(self.embd))

        if not self.files[self.split]:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.images_base))

        print("Found %d %s images" % (len(self.files[self.split]), self.split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        lbl_path = self.files[self.split][index].rstrip()
        img_path = os.path.join(
            self.images_base,
            os.path.basename(lbl_path))[:-3]+'jpg'

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        if len(img.shape)==2:
            img = img[:,:, np.newaxis]
            img = img.repeat(3, axis=2)

        h,w,c = img.shape
        lbl = m.imread(lbl_path)
        lbl = cv2.resize(lbl,(w,h), interpolation = cv2.INTER_NEAREST)

        lbl = self.encode_valid(lbl)

        if self.mode=='train':
            lbl = self.encode_seen(lbl)

        '''import cv2
        cv2.namedWindow("Image")
        cv2.imshow("Image", self.decode_segmap(lbl))
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl

    def encode_valid(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def encode_seen(self, mask):
        if self.transductive==1:
            for _unseenc in self.unseen_classes:
                mask[mask == _unseenc] = self.ignore_index
        elif self.check_unseen(mask):
            mask[:] = self.ignore_index
        return mask

    def check_unseen(self,mask):
        unique_class = np.unique(mask)
        has_unseen_class = False
        for u_class in unique_class:
            if u_class in self.unseen_classes:
                has_unseen_class =True
        return has_unseen_class


    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

