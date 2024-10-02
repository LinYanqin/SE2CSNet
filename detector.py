'''
-*- coding:utf-8 -*-
@File:detector.py
'''
import os
import h5py
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import torch
import scipy.io as sio
from dataset import SpinEchoDataSet
from network import SpinEchoNet

device = torch.device('cpu')
class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.net = SpinEchoNet().to(device)
        self.weight_path = 'params/net.pt'
        if os.path.exists(self.weight_path):
            self.net.load_state_dict(torch.load(self.weight_path, map_location=device))

        self.net.eval()

    def forward(self, input):
        output = self.net(input)
        return output

    def detect(self, data_path, save_path):
        # get data
        a = h5py.File(data_path)
        data_all = a['data']
        data = np.array(data_all['data_x']).reshape(1, 1, 4096, 8).astype('float32')
        data = torch.from_numpy(data).to(device)

        with torch.no_grad():
            results, per_out = self.net(data)
        results = (results - results.min()) / (results.max() - results.min())
        results_cpu = results.cpu().detach().numpy()

        sio.savemat(save_path, {'x': data.reshape(4096, 8).tolist(),
                                'pre': results_cpu.reshape(1, 4096).tolist(),
                                'per0': per_out[0].reshape(128, 4096).tolist()
                                })

        print("Detect successfully!")

if __name__ == '__main__':
    # exp = 'exp/exp_azithromycin.mat'
    # exp = 'exp/exp_asarone.mat'
    exp = 'exp/exp_estradiol.mat'
    # exp = 'exp/exp_mixture.mat'

    detector = Detector()
    detector.detect(exp, 'predict/pre.mat')
