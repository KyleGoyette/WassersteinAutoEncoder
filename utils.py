import numpy as np

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle
import config

def train(model, dloader, loss_type='VAE'):
    for batch_index, (data,_) in dloader:

        data = Variable(data).cuda()

def load_data_mnist(batch_size=config.batch_size, test=False):
    train_path = '/data/lisa/data/mnist/mnist-python/train.pkl'
    if not os.path.isfile(train_path):
        raise Exception('Incorrect path to data')

    else:
        
        train_data = pickle.load(open(train_path,'rb'),encoding='latin1')
        train_dataset = TensorDataset(torch.Tensor(train_data['data']), torch.IntTensor(train_data['labels']))
        trainloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        if (test):
            test_data = pickle.load(open(test_path,'rb'), encoding='latin1')

            test_dataset = TensorDataset(torch.Tensor(test_data['data']), torch.IntTensor(test_data['labels']))
            testloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
            return trainloader, testloader

        return trainloader, None