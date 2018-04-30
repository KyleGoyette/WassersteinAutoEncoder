import torch
import torch.nn
import torch.nn.functional as F
import os
import pickle
import config

def train(model, dloader, loss_type='VAE'):
    for batch_index, (data,_) in dloader:

        data = Variable(data).cuda()

def load_data_mnist(batch_size=config.batch_size, test=False):
    train_path = '/data/lisa/data/mnist/mnist-python/train.pkl'
    if not os.path.isfile(path):
        raise Exception('Incorrect path to data')

    else:
        train_data = pickle.load(open(train_path,'rb'))
        train_data = np.array(train_data)
        
        train_data.shuffle()

        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_data))
        trainloader = torch.utils.data.Dataloader(train_dataset,batch_size=batch_size,shuffle=True)
        if (test):
            test_data = pickle.load(open(test_path,'rb'))
            test_data = np.array(test_data)
            test_data.shuffle()
            test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_data))
            testloader = torch.utils.data.Dataloader(test_dataset,batch_size=batch_size,shuffle=True)
            return trainloader, testloader

        return trainloader