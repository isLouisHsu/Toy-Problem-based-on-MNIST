# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-07-11 11:15:04
@LastEditTime: 2019-08-16 13:38:35
@Update: 
'''
import os
import cv2
import struct
import numpy as np

import torch
from torch.utils.data import Dataset

class MNIST(Dataset):

    def __init__(self, mode='train', used_labels=None):
        if mode == 'train':
            images_path = '../data/train-images.idx3-ubyte'
            labels_path = '../data/train-labels.idx1-ubyte'
        elif mode == 'valid':
            images_path = '../data/t10k-images.idx3-ubyte'
            labels_path = '../data/t10k-labels.idx1-ubyte'
        
        with open(images_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            loaded = np.fromfile(images_path, dtype = np.uint8)
            self.images = loaded[16:].reshape(num, 28, 28).astype(np.float)
        
        with open(labels_path, 'rb') as f:
            magic, n = struct.unpack('>II', f.read(8))
            self.labels = np.fromfile(f, dtype = np.uint8).astype(np.float)

        if used_labels is not None:
            selected = np.zeros_like(self.labels, dtype=np.bool)
            for label in used_labels:
                selected = np.bitwise_or(selected, (self.labels == label))

            self.images = self.images[selected]
            self.labels = self.labels[selected]
        
        labels_set = set(list(self.labels))
        labels_dict = dict(zip(labels_set, range(len(labels_set))))
        for k, v in labels_dict.items():
            self.labels[self.labels == k] = v
            
        self.n_classes = len(labels_set)

    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]

        image = torch.from_numpy(image).unsqueeze(0).float() / 255.
        
        return image, label

    def __len__(self):

        return len(self.images)

if __name__ == "__main__":
    
    D = MNIST('train', [1, 2])
    for i in range(10):
        X, y = D[i]
        
        cv2.imshow("%d" % y, (X[0]*255).numpy().astype('uint8'))
        cv2.waitKey(500)