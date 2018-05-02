from utils import load_data_mnist, load_data_celeba,train,test,save_model, pretrain
import torch
import VAE
import WAE
import config
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="mnist_vae", help="mnist_vae, mnist_wae_mmd, mnist_wae_gan,celeba...")

FLAGS = parser.parse_args()

if FLAGS.exp == "mnist_vae":
    confs = config.conf_mnist_vae
    model = VAE.VAE(confs)
    optimizer = torch.optim.Adam(model.myparameters,lr = confs['lr'], betas=(confs['B1'],confs['B2']))
    scheduler1 = MultiStepLR(optimizer, milestones= confs['milestones1'],gamma=0.5)
    scheduler2 = MultiStepLR(optimizer, milestones= confs['milestones2'],gamma=0.2)
elif FLAGS.exp == 'celeba_vae':
    confs = config.conf_celeba_vae
    model = VAE.VAE(confs)
    optimizer = torch.optim.Adam(model.myparameters,lr = confs['lr'], betas=(confs['B1'],confs['B2']))
    scheduler1 = MultiStepLR(optimizer, milestones= confs['milestones1'],gamma=0.5)
    scheduler2 = MultiStepLR(optimizer, milestones= confs['milestones2'],gamma=0.2)
elif FLAGS.exp == 'mnist_waegan':
    confs = config.conf_mnist_wae_gan
    model = WAE.WAE_GAN(confs)
    optimizer_wae = torch.optim.Adam(model.myparameters,lr = confs['lr'], betas=(confs['B1'],confs['B2']))
    optimizer_disc = torch.optim.Adam(model.discriminator.parameters(),lr = confs['lr_disc'], betas=(confs['B1_disc'],confs['B2_disc']))
    scheduler1 = MultiStepLR(optimizer_wae, milestones= confs['milestones1'],gamma=0.5)
    scheduler2 = MultiStepLR(optimizer_wae, milestones= confs['milestones2'],gamma=0.2)
    scheduler1_disc = MultiStepLR(optimizer_disc, milestones= confs['milestones1'],gamma=0.5)
    scheduler2_disc = MultiStepLR(optimizer_disc, milestones= confs['milestones2'],gamma=0.2)
    optimizer = (optimizer_wae, optimizer_disc)

elif FLAGS.exp == 'celeba_waegan':
    confs = config.conf_celeba_wae_gan
    model = WAE.WAE_GAN(confs)
    optimizer_wae = torch.optim.Adam(model.myparameters,lr = confs['lr'], betas=(confs['B1'],confs['B2']))
    optimizer_disc = torch.optim.Adam(model.myparameters,lr = confs['lr_disc'], betas=(confs['B1_disc'],confs['B2_disc']))
    scheduler1 = MultiStepLR(optimizer_wae, milestones= confs['milestones1'],gamma=0.5)
    scheduler2 = MultiStepLR(optimizer_wae, milestones= confs['milestones2'],gamma=0.2)
    scheduler1_disc = MultiStepLR(optimizer_disc, milestones= confs['milestones1'],gamma=0.5)
    scheduler2_disc = MultiStepLR(optimizer_disc, milestones= confs['milestones2'],gamma=0.2)
    optimizer = (optimizer_wae, optimizer_disc)

elif FLAGS.exp == 'mnist_waemmd':
    confs = config.conf_mnist_wae_mmd
    model = WAE.WAE_MMD(confs)
    optimizer = torch.optim.Adam(model.myparameters,lr = confs['lr'], betas=(confs['B1'],confs['B2']))
    scheduler1 = MultiStepLR(optimizer, milestones= confs['milestones1'],gamma=0.5)
    scheduler2 = MultiStepLR(optimizer, milestones= confs['milestones2'],gamma=0.2)

elif FLAGS.exp == 'celeba_waemmd':
    confs = config.conf_celeba_wae_mmd
    model = WAE.WAE_MMD(confs)
    optimizer = torch.optim.Adam(model.myparameters,lr = confs['lr'], betas=(confs['B1'],confs['B2']))
    scheduler1 = MultiStepLR(optimizer, milestones= confs['milestones1'],gamma=0.5)
    scheduler2 = MultiStepLR(optimizer, milestones= confs['milestones2'],gamma=0.2)



if confs['dataset']== 'MNIST':
    trainloader, testloader = load_data_mnist(test=True)
elif confs['dataset'] == 'celeba':
    trainloader, testloader = load_data_celeba()

if confs['CUDA']:
    model.cuda()



#if confs['pretrain']:
#    optimizer_p = torch.optim.Adam(model.encoder.parameters(),lr = confs['lr'], betas=(confs['B1_disc'],confs['B2_disc']))
#    pretrain(model,trainloader,optimizer_p,confs)

train_losses = []
test_losses = []
print('Beginning training...')
for epoch in range(confs['NUMEPOCHS']):
    scheduler1.step()
    scheduler2.step()
    if confs['loss'] == 'wae-gan':
        scheduler1_disc.step()
        scheduler2_disc.step()
    train_loss = train(model,trainloader,optimizer,confs,epoch)
    train_losses.append(train_loss)
    if testloader != None:
        test_loss = test(model,testloader,epoch,confs,add_noise=confs['noise'])
        test_losses.append(test_loss)
        print('Epoch: {} Train Loss: {} Test Loss: {}'.format(epoch,train_loss,test_loss))
    else:
        print('Epoch: {} Train Loss: {}'.format(epoch,train_loss))
    if epoch % config.SAVEFREQ == 0:
        save_model(confs['dataset']+'_'+confs['type'],model,epoch,confs)
save_model(confs['dataset'],model,epoch,confs)
np.save('./losses/{}_train'.format(FLAGS.exp),train_losses)
np.save('./losses/{}_test'.format(FLAGS.exp),test_losses)