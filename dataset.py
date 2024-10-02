'''
-*- coding:utf-8 -*-
@File:dataset.py
'''
from torch.utils.data import Dataset
import torch

class SpinEchoDataSet(Dataset):
    def __init__(self):
        # open data file and label file
        with open('./dataset/label/label.txt', 'r') as flabel:
            self.labelset = flabel.readlines()
        with open('./dataset/data/data.txt', 'r') as fdata:
            self.dataset = fdata.readlines()

        # dataset size
        self.dataset_size = len(self.dataset) // 8

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        data = self.dataset[(index * 8):((index + 1) * 8)]  # 8*4096, list[str]
        label = self.labelset[(index * 1):((index + 1) * 1)]  # 1*4096, list[str]
        temp_data = [i.strip('\n').split('\t') for i in data]  # 8*4096, list[list[str]]
        temp_label = [i.strip('\n').split('\t') for i in label]  # 1*4096, list[list[str]]
        for i in range(len(temp_data[0])):
            for j in range(len(temp_data)):
                try:
                    temp_data[j][i] = float(temp_data[j][i])  # 8*4096, list[list[int]]
                except ValueError:
                    temp_data[j][i] = 0.0
            for k in range(len(temp_label)):
                try:
                    temp_label[k][i] = float(temp_label[k][i])  # 1*4096, list[list[int]]
                except ValueError:
                    temp_label[k][i] = 0.0

        data = torch.Tensor(temp_data)
        data = data.unsqueeze(-1).transpose(0, 2)  # 1*4096*8, tensor
        label = torch.Tensor(temp_label)
        label = label.unsqueeze(-1)  # 1*4096*1, tensor
        return data, label

    def get_sample_size(self, index):
        sample_data, sample_label = self.__getitem__(index)
        return sample_data.size(), sample_label.size()