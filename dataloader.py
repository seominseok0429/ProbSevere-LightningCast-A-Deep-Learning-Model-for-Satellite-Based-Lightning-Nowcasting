from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import glob


class GLMLoader(Dataset):
    """
    is_train - 0 train, 1 val
    # /workspace/SSD_4T_b/GK2A/lighting/dataset/2022/20220828
    """
    def __init__(self, is_train=0):
        self.ch2_path = glob.glob('/workspace/SSD_4T_b/GK2A/lighting/dataset/2022/20220828/CH02*')

    def __len__(self):
        return len(self.ch2_path)

    def __getitem__(self, idx):
        ch2 = np.load(self.ch2_path[idx])[:, :, 0]
        ch5 = np.load(self.ch2_path[idx].replace('CH02', 'CH05'))[:, :, 0]
        ch13 = np.load(self.ch2_path[idx].replace('CH02', 'CH13'))[:, :, 0]
        ch15 = np.load(self.ch2_path[idx].replace('CH02', 'CH15'))[:, :, 0]

        gt = np.load(self.ch2_path[idx].replace('CH02', 'FED'))[:, :, 0]
        gt = gt[np.newaxis, :, :]

        ch5 = cv2.resize(ch5, (1920, 1920))
        ch13 = cv2.resize(ch13, (1920, 1920))
        ch15 = cv2.resize(ch15, (1920, 1920))

        concat_result = np.stack([ch2, ch5, ch13, ch15], axis=0) 
        return concat_result/255.,  gt/255.

if __name__ == '__main__':
    loader = GLMLoader()
    inputs, gt = loader[0]
    # torch.Size([1920, 4, 1920]) torch.Size([480, 1, 480])
    # but i wish outputd for [channels, width, hight]

