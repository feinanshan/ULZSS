import os
import pdb
import torch
import numpy as np
import scipy.misc as m
import cv2
from torch.utils import data
import torchvision.transforms.functional as TF
from ptsemseg.utils import recursive_glob
import scipy.io as scio
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale


def color_map(N=60, normalized=False):
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

class context460Loader(data.Dataset):

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
        if mode == 'train':
            self.split = cfg["train_split"]
        elif mode == 'val':
            self.split = cfg["val_split"]

        self.files = {}
        self.images_base = os.path.join(self.root, "Img", self.split)
        self.annotations_base = os.path.join(self.root, "trainval400", self.split)

        self.files[self.split] = recursive_glob(rootdir=self.annotations_base, suffix=".mat")

        self.n_classes = 215
        self.void_classes = [0, 1, 3, 4, 5, 7, 12, 13, 14, 16, 
                            20, 21, 24, 29, 35, 38, 41, 47, 50, 52, 
                            54, 63, 64, 67, 71, 73, 74, 76, 77, 79, 
                            81, 82, 83, 84, 89, 91, 92, 93, 94, 95, 
                            97, 99, 100, 101, 102, 103, 107, 108, 109, 111, 
                            112, 114, 116, 117, 118, 119, 120, 121, 125, 126, 
                            127, 129, 130, 131, 132, 133, 134, 135, 137, 139, 
                            142, 143, 145, 146, 147, 151, 152, 153, 156, 157, 
                            160, 161, 163, 164, 166, 167, 168, 171, 172, 173, 
                            174, 175, 177, 178, 179, 180, 182, 183, 188, 192, 
                            193, 197, 198, 200, 201, 202, 203, 205, 206, 209, 
                            210, 212, 214, 215, 217, 218, 222, 224, 226, 227, 
                            229, 230, 231, 233, 234, 235, 236, 237, 238, 239, 
                            240, 241, 242, 243, 245, 246, 249, 253, 254, 255, 
                            256, 257, 264, 267, 270, 274, 276, 278, 279, 280, 
                            283, 285, 288, 292, 298, 299, 300, 301, 302, 304, 
                            305, 310, 312, 313, 315, 317, 318, 321, 322, 325, 
                            327, 328, 331, 332, 335, 336, 337, 338, 339, 340, 
                            341, 343, 344, 345, 346, 348, 351, 352, 353, 358, 
                            362, 364, 365, 367, 369, 370, 372, 375, 376, 379, 
                            380, 381, 382, 385, 386, 387, 388, 389, 390, 391, 
                            392, 393, 394, 395, 396, 398, 399, 401, 404, 407, 
                            408, 409, 411, 414, 417, 421, 422, 423, 425, 426, 
                            428, 429, 433, 439, 441, 442, 444, 447, 448, 449, 
                            450, 451, 453, 455, 459]

        self.valid_classes = [2, 6, 8, 9, 10, 11, 15, 17, 18, 19, 
                              22, 23, 25, 26, 27, 28, 30, 31, 32, 33, 
                              34, 36, 37, 39, 40, 42, 43, 44, 45, 46, 
                              48, 49, 51, 53, 55, 56, 57, 58, 59, 60, 
                              61, 62, 65, 66, 68, 69, 70, 72, 75, 78, 
                              80, 85, 86, 87, 88, 90, 96, 98, 104, 105, 
                              106, 110, 113, 115, 122, 123, 124, 128, 136, 138, 
                              140, 141, 144, 148, 149, 150, 154, 155, 158, 159, 
                              162, 165, 169, 170, 176, 181, 184, 185, 186, 187, 
                              189, 190, 191, 194, 195, 196, 199, 204, 207, 208, 
                              211, 213, 216, 219, 220, 221, 223, 225, 228, 232, 
                              244, 247, 248, 250, 251, 252, 258, 259, 260, 261, 
                              262, 263, 265, 266, 268, 269, 271, 272, 273, 275, 
                              277, 281, 282, 284, 286, 287, 289, 290, 291, 293, 
                              294, 295, 296, 297, 303, 306, 307, 308, 309, 311, 
                              314, 316, 319, 320, 323, 324, 326, 329, 330, 333, 
                              334, 342, 347, 349, 350, 354, 355, 356, 357, 359, 
                              360, 361, 363, 366, 368, 371, 373, 374, 377, 378, 
                              383, 384, 397, 400, 402, 403, 405, 406, 410, 412, 
                              413, 415, 416, 418, 419, 420, 424, 427, 430, 431, 
                              432, 434, 435, 436, 437, 438, 440, 443, 445, 446, 
                              452, 454, 456, 457, 458]

        self.unseen_classes = [1, 2, 4, 5, 6, 7, 13, 14, 15, 16, 
                              18, 21, 22, 23, 24, 25, 26, 30, 31, 32, 
                              33, 34, 35, 36, 37, 39, 40, 41, 43, 45, 
                              46, 48, 49, 52, 53, 54, 55, 56, 60, 61, 
                              64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 
                              75, 76, 77, 81, 82, 83, 84, 85, 86, 87, 
                              88, 91, 92, 93, 94, 95, 96, 97, 99, 100, 
                              101, 102, 103, 105, 106, 107, 108, 110, 111, 112, 
                              113, 114, 115, 119, 120, 121, 122, 123, 124, 125, 
                              126, 127, 128, 129, 130, 131, 132, 134, 135, 136, 
                              137, 138, 139, 140, 143, 144, 145, 146, 148, 149, 
                              150, 151, 152, 153, 154, 157, 158, 159, 160, 161, 
                              164, 167, 168, 169, 171, 172, 175, 176, 177, 178, 
                              179, 180, 181, 183, 184, 185, 186, 187, 188, 189, 
                              190, 193, 194, 198, 199, 200, 201, 202, 203, 204, 
                              205, 207, 209, 210, 212, 213] #Note that these should be choosen from IDs for valid categories

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(215)))

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
            os.path.basename(lbl_path))[:-3] + 'jpg'

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        h, w, c = img.shape

        lbl = scio.loadmat(lbl_path)["LabelMap"]
        #lbl = m.imread(lbl_path)
        lbl = cv2.resize(lbl, (w, h), interpolation=cv2.INTER_NEAREST)

        lbl = self.encode_valid(lbl)

        if self.mode == 'train':
            lbl = self.encode_seen(lbl)

        '''import cv2
        cv2.namedWindow("Image")
        cv2.imshow("Image", self.decode_segmap(lbl))
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl.astype(np.uint8))

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        
        return img, lbl

    def encode_valid(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = 500
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        mask[mask==500] = self.ignore_index

        return mask

    def encode_seen(self, mask):
        if self.transductive == 1:
            for _unseenc in self.unseen_classes:
                mask[mask == _unseenc] = self.ignore_index
        elif self.check_unseen(mask):
            mask[:] = self.ignore_index
        return mask

    def check_unseen(self, mask):
        unique_class = np.unique(mask)
        has_unseen_class = False
        for u_class in unique_class:
            if u_class in self.unseen_classes:
                has_unseen_class = True
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

