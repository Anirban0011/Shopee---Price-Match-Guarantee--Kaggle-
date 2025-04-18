import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class SHOPEETextDataset(Dataset):
    def __init__(self, df):

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        text = row.title
        return text
