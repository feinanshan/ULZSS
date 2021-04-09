import os
import torch
import numpy as np
import scipy.misc as m
import scipy.io as scio
from torch.utils import data
from ptsemseg.utils import recursive_glob


class cityscapesLoader(data.Dataset):

    colors = [  # [  0,   0,   0],
        [230, 0, 0],#
        [128, 64, 128],
        [244, 35, 232],
        [220, 0, 220],#
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [0, 220, 220],#
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [180, 230, 20],#
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
    label_colours = dict(zip(range(23), colors))



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
        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)
        self.files[self.split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.n_classes = 23
        self.void_classes = [ 0, 1,  2,  3,  4, 
                              5, 9, 14, 16, 18, 
                             29, -1]
        self.valid_classes = [6,  7,  8, 10, 11, 
                             12, 13, 15, 17, 19, 
                             20, 21, 22, 23, 24, 
                             25, 26, 27, 28, 30, 
                             31, 32, 33]

        self.unseen12 = [0, 1, 4, 7, 8, 10, 12, 13, 15, 17, 18, 22]
        self.unseen8 = [7, 8, 10, 12, 17, 18, 22]
        self.unseen4 = [7, 10, 17, 22]
        self.unseen2 = [17, 22]

        if cfg["unseen"] ==0:
            self.unseen_classes = [] # Note that these should be choosen from IDs for valid categories
        elif cfg["unseen"] ==2:
            self.unseen_classes = self.unseen2  # Note that these should be choosen from IDs for valid categories
        elif cfg["unseen"] ==4:
            self.unseen_classes = self.unseen4  # Note that these should be choosen from IDs for valid categories
        elif cfg["unseen"]==8:
            self.unseen_classes = self.unseen8  # Note that these should be choosen from IDs for valid categories
        elif cfg["unseen"]==12:
            self.unseen_classes = self.unseen12  # Note that these should be choosen from IDs for valid categories
        else:
            raise Exception("Only support 4, 8, 12 unseen")

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(23)))

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
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)

        lbl = self.encode_valid(lbl)

        if self.mode=='train':
            lbl = self.encode_seen(lbl)

        '''import cv2
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
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