import numpy as np

import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle
import config
CUDA = config.CUDA
def train_VAE(model, dloader,optimizer,num_epochs = config.NUMEPOCHS,add_noise=True):
    for epoch in config.NUMEPOCHS:
        for batch_index, (data,_) in enumerate(dloader):  
            if CUDA:
                data = data.cuda()
            if add_noise:
                noise = torch.FloatTensor(torch.zeros(data.shape)).normal_()
                noise = truncate_noise(noise)
                if CUDA:
                    noise = noise.cuda()

                data += noise
            data = Variable(data)
            if config.dataset == 'MNIST':
                data = data.view(-1,1,28,28)
            elif config.dataset == 'CelebA':
                data = data.view(-1,3,64,64)
            optimizer.zero_grad()
            mu, logvar = model.encode(data)
            z = model.reparameterize(mu,logvar)
            recon_x = model.decode(z)
            loss,recon_loss, KLD_loss = model.loss(recon_x,data,mu,logvar)
            loss.backward()
            optimizer.step()
        
        save_model(name,model,epoch) 


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

def truncate_noise(noise):
    return (100*noise).round()/100


def save_model(name,model,epoch):
    torch.save(model.state_dict(),"/data/milatmp1/goyettky/IFT6135/Project/WassersteinAutoEncoder/models/{}_{}_{}.pt".format(name,epoch))
    
def load_model(fname,model):
    loaded_state_dict = torch.load("/data/milatmp1/goyettky/IFT6135/Project/WassersteinAutoEncoder/models/{}_{}_{}.pt".format(fname))
    state_dict = model.state_dict()
    state_dict.update(loaded_state_dict)
    model.load_state_dict(loaded_state_dict)