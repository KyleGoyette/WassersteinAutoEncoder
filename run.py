from utils import load_data_mnist,train,test,save_model
import torch
import VAE
import config
trainloader, testloader = load_data_mnist()
model = VAE.VAE(config.conf_mnist)

confs = config.conf_mnist
if config.conf_mnist['CUDA']==True:
    model.cuda()
optimizer = torch.optim.Adam(model.myparameters,lr = confs['lr'], betas=(confs['B1'],confs['B2']))

train_losses = []
test_losses = []

for epoch in range(config.NUMEPOCHS):
    train_loss = train(model,trainloader,optimizer,epoch)
    train_losses.append(train_loss)
    if testloader != None:
        test_loss = test(model,testloader,epoch)
        test_losses.append(test_loss)
    print('Epoch: {} Train Loss: {} Test Loss: {}'.format(epoch,train_loss,test_loss))
    if epoch % config.SAVEFREQ == 0:
        save_model(confs['dataset'],model,epoch)