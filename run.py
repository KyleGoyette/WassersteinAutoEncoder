from utils import load_data_mnist,train,test,save_model
import torch
import VAE
import config
from torch.optim.lr_scheduler import MultiStepLR
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="mnist_vae", help="mnist_vae, mnist_wae_mmd, mnist_wae_gan,celeba...")


FLAGS = parser.parse_args()

if FLAGS.exp == "mnist_vae":
    confs = config.conf_mnist_vae

    model = VAE.VAE(confs)
    optimizer = torch.optim.Adam(model.myparameters,lr = confs['lr'], betas=(confs['B1'],confs['B2']))
    scheduler1 = MultiStepLR(optimizer, milestones= confs['milestones1'],gamma=0.5)
    scheduler2 = MultiStepLR(optimizer, milestones= confs['milestones2'],gamma=0.2)
elif FLAFS.exp == 'celeba_vae':
    confs = config.conf_celeba_vae
    model = VAE.VAE(confs)
    optimizer = torch.optim.Adam(model.myparameters,lr = confs['lr'], betas=(confs['B1'],confs['B2']))
    scheduler1 = MultiStepLR(optimizer, milestones= confs['milestones1'],gamma=0.5)
    scheduler2 = MultiStepLR(optimizer, milestones= confs['milestones2'],gamma=0.2)

if confs['dataset']== 'MNIST':
    trainloader, testloader = load_data_mnist()
elif confs['dataset'] == 'celeba':
    trainloader, testloader = load_data_celeba()

if confs['CUDA']:
    model.cuda()

train_losses = []
test_losses = []

for epoch in range(confs['NUMEPOCHS']):
    scheduler1.step()
    scheduler2.step()
    train_loss = train(model,trainloader,optimizer,epoch)
    train_losses.append(train_loss)
    if testloader != None:
        test_loss = test(model,testloader,epoch,add_noise=confs['noise'])
        test_losses.append(test_loss)
        print('Epoch: {} Train Loss: {} Test Loss: {}'.format(epoch,train_loss,test_loss))
    else:
        print('Epoch: {} Train Loss: {}'.format(epoch,train_loss))
    if epoch % config.SAVEFREQ == 0:
        save_model(confs['dataset'],model,epoch)
save_model(confs['dataset'],model,epoch)