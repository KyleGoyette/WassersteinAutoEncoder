from utils import load_data_mnist,train,test
import torch
import VAE
import config
trainloader, testloader = load_data_mnist()
model = VAE.VAE(config.conf_mnist)

if config.conf_mnist['CUDA']==True:
    model.cuda()
optimizer = torch.optim.Adam(model.myparameters)
for epoch in range(50):
    train_loss = train(model,trainloader,optimizer,epoch)
    #test_loss = test(model,testloader,epoch)