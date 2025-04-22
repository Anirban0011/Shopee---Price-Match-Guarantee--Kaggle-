import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class SHOPEEImageDataset(Dataset):
    def __init__(self, df, dir, transform=None):

        self.df = df
        self.dir = dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        img = cv2.imread(f"{dir}/{row.image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # CV2 default BGR
        img = img.copy()

        if self.transform is not None:
            img = self.transform(image = img) # albu compatible
            img = img['image']

        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1) # pytorch ready NCHW

        return torch.tensor(img).float(), torch.tensor(row.label_group).float()
