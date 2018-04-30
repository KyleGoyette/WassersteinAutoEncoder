from utils import load_data_mnist,train_VAE
import torch
import VAE
trainloader, testloader = load_data_mnist()
model = VAE.VAE()

#model.cuda()
optimizer = torch.optim.Adam(model.myparameters)
train_VAE(model,trainloader,optimizer)
