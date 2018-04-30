import torch
import torch.nn
import torch.nn.functional as F
import os
import pickle
import config

def train(model, dloader, loss_type='VAE'):
    for batch_index, (data,_) in dloader:

        data = Variable(data).cuda()

def load_data_mnist(batch_size=config.batch_size,test_split=0.2):
    path = '/data/lisa/data/mnist/mnist.pkl'
    if not os.path.isfile(data):
        raise Exception('Incorrect path to data')

    else:
        data = pickle.load(open(path),'rb')
        data = np.array(data)
        
        n = data.shape[0]
        data.shuffle()
        train_data = data[:n*int(1-test_split)]
        
        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_data))
        trainloader = torch.utils.data.Dataloader(train_dataset,batch_size=batch_size,shuffle=True)
        if (test_split != 0)
            test_data = data[n*int(1-test_split):]
            test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_data))
            testloader = torch.utils.data.Dataloader(test_dataset,batch_size=batch_size,shuffle=True)
            return trainloader, testloader

        return trainloader