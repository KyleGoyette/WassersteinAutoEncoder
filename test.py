from utils import load_data_mnist,train
import torch
import VAE
import config
trainloader, testloader = load_data_mnist()
model = VAE.VAE(config.conf_mnist)

if config.conf_mnist['CUDA']==True:
    model.cuda()
optimizer = torch.optim.Adam(model.myparameters)
train(model,trainloader,optimizer)
