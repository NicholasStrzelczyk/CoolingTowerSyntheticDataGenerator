import cv2
import numpy as np
from torch.utils.data import Dataset


class CustomDS(Dataset):
    def __init__(self, x_set, y_set, resize_shape=(1024, 1024), img_transpose=None):
        self.img_transpose = img_transpose  # (2, 0, 1) for rgb
        self.resize_shape = resize_shape
        self.x = x_set
        self.y = y_set

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        target = cv2.imread(self.y[idx], cv2.IMREAD_COLOR)
        image, target = self.preprocess(image, target)
        return image, target

    def preprocess(self, img, tgt):
        if self.resize_shape is not None:
            img = cv2.resize(img, self.resize_shape)
            tgt = cv2.resize(tgt, self.resize_shape)
        if self.img_transpose is not None:
            img = np.transpose(img, axes=self.img_transpose)
            tgt = np.transpose(tgt, axes=self.img_transpose)
        return img, tgt
