import os
import cv2
import struct
import numpy as np

import torch
from torch.utils.data import Dataset

class MNIST(Dataset):

    def __init__(self, mode='train'):
        if mode == 'train':
            images_path = './data/train-images.idx3-ubyte'
            labels_path = './data/train-labels.idx1-ubyte'
        elif mode == 'valid':
            images_path = './data/t10k-images.idx3-ubyte'
            labels_path = './data/t10k-labels.idx1-ubyte'
        
        with open(images_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            loaded = np.fromfile(images_path, dtype = np.uint8)
            self.images = loaded[16:].reshape(num, 28, 28).astype(np.float)
        
        with open(labels_path, 'rb') as f:
            magic, n = struct.unpack('>II', f.read(8))
            self.labels = np.fromfile(f, dtype = np.uint8).astype(np.float)

    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]

        image = torch.from_numpy(image).unsqueeze(0).float() / 255.
        
        return image, label

    def __len__(self):

        return len(self.images)
        