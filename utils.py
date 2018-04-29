import torch
import torch.nn
import torch.nn.functional as F
import os

def train(model, dloader, loss_type='VAE'):
    for batch_index, (data,_) in dloader:

        data = Variable(data).cuda()

def load_data(name, batch_size):
    if not os.path.isfile(name):
        raise Exception('Incorrect path to data')

    else:
        dataloader = torch.utils.data.Dataloader(dataset,batch_size=100,shuffle=True)
        return dataloader