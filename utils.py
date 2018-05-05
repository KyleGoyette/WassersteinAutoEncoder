import numpy as np

import torch
import torch.nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle
import config
from PIL import Image
import matplotlib.pyplot as plt

from torchvision.utils import save_image

def train(model, dloader,optimizer,confs, epoch,add_noise=True):
    if confs['loss'] == 'wae-gan':
        optimizer, optimizer_disc = optimizer
    model.encoder.train()
    model.decoder.train()
    model.train()
    train_loss = 0
    recon_losses = 0
    match_losses = 0
    for batch_index, (data,_) in enumerate(dloader):  
        if confs['CUDA']:
            data = data.cuda()
        
        if confs['dataset'] == 'MNIST':
            data = data.view(-1,1,28,28)
        elif confs['dataset'] == 'CelebA':
            data = data.view(-1,3,64,64)
        orig_data = data.clone()
        if add_noise:
            noise = torch.FloatTensor(torch.zeros(data.shape)).normal_()
            noise = truncate_noise(noise)
            if confs['CUDA']:
                noise = noise.cuda()

            data += noise
        data = Variable(data)
        
        if confs['loss'] == 'vae':
            optimizer.zero_grad()
            recon_x, mu, logvar = model.forward(data)
            tot_loss,recon_loss, KLD_loss = model.loss(recon_x,Variable(orig_data),mu,logvar)

        
            tot_loss.backward()
            optimizer.step()
            train_loss += tot_loss.data[0]
            recon_losses += recon_loss.data[0]
            match_losses += KLD_loss.data[0]
        
        elif confs['loss'] == 'wae-gan':
            optimizer_disc.zero_grad()
            optimizer.zero_grad()
            mu, logvar = model.encode(data)
            z_tilde = model.reparameterize(mu,logvar)
            z = torch.autograd.Variable(torch.normal(logvar.shape),std=confs['sig_z']).cuda()

            recon_x = model.decode(z_tilde)

            loss, d_loss,recon_loss,match_loss = model.loss(recon_x,data,z,z_tilde)
            d_loss.backward(retain_graph=True)
            optimizer_disc.step()

            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            recon_losses += recon_loss.data[0]
            match_losses += match_loss.data[0]
        elif confs['loss'] == 'wae-mmd':
            optimizer.zero_grad()
            mu, logvar = model.encode(data)
            z_tilde = model.reparameterize(mu,logvar)
            z = torch.autograd.Variable(torch.normal(logvar.shape),std=confs['sig_z']).cuda()

            recon_x = model.decode(z_tilde)
            loss,recon_loss,match_loss = model.loss(recon_x,data,z,z_tilde)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            recon_losses += recon_loss.data[0]
            match_losses += match_loss.data[0]
    
    
    return train_loss/len(dloader.dataset), recon_losses/len(dloader.dataset), match_losses/len(dloader.dataset)

def test(model,dloader,epoch,confs,add_noise=True):
    model.encoder.eval()
    model.decoder.eval()
    model.eval()
    test_loss = 0
    recon_losses = 0
    match_losses = 0
    for batch_index, (data, _) in enumerate(dloader):
        if confs['CUDA']: 
            data = data.cuda()
        if confs['dataset'] == 'MNIST':
            data = data.view(-1,1,28,28)
        elif confs['dataset'] == 'CelebA':
            data = data.view(-1,3,64,64)
        orig_data = data.clone()
        if add_noise:
            noise = torch.FloatTensor(torch.zeros(data.shape)).normal_()
            noise = truncate_noise(noise)
            if confs['CUDA']:
                noise = noise.cuda()
            data += noise
        data = Variable(data)

        
        if confs['loss'] == 'vae':
            recon_x, mu, logvar = model.forward(data)
            loss, bce_loss, KLD_loss = model.loss(recon_x,Variable(orig_data),mu,logvar)
            recon_losses += bce_loss.data[0]
            match_losses += KLD_loss.data[0]
        elif confs['loss'] == 'wae-gan':
            mu, logvar = model.encode(data)
            z_tilde = model.reparameterize(mu,logvar)
            z = torch.autograd.Variable(torch.normal(logvar.shape),std=confs['sig_z']).cuda()

            recon_x = model.decode(z_tilde)

            loss, d_loss, recon_loss, match_loss = model.loss(recon_x,data,z,z_tilde)
            recon_losses += recon_loss.data[0]
            match_losses += match_loss.data[0]
        elif confs['loss'] == 'wae-mmd':
            mu, logvar = model.encode(data)
            z_tilde = model.reparameterize(mu,logvar)
            z = torch.autograd.Variable(torch.normal(logvar.shape),std=confs['sig_z']).cuda()
            recon_x = model.decode(z_tilde)

            loss,recon_loss,match_loss = model.loss(recon_x,data,z,z_tilde)
            recon_losses += recon_loss.data[0]
            match_losses += match_loss.data[0]
        test_loss += loss.data[0]

    if (epoch%config.REPORTFREQ == 0) or epoch == confs['NUMEPOCHS']:
        save_images(recon_x,orig_data,epoch,confs)
    return test_loss/len(dloader.dataset), recon_losses/len(dloader.dataset), match_losses/len(dloader.dataset)

def pretrain(model,train_loader,optimizer,confs):
    for batch_index, (data, _) in train_loader:

        if confs['CUDA']:
            data = data.cuda()
        
        if confs['dataset'] == 'MNIST':
            data = data.view(-1,1,28,28)
        elif confs['dataset'] == 'CelebA':
            data = data.view(-1,3,64,64)
        orig_data = data.clone()
        if confs['noise']:
            noise = torch.FloatTensor(torch.zeros(data.shape)).normal_()
            noise = truncate_noise(noise)
            if confs['CUDA']:
                noise = noise.cuda()

            data += noise
        data = Variable(data)
        mu, logvar = model.encode(data)
        loss = model.pretain_loss(mu,logvar)
        loss.backward()
        optimizer.step()

        print(loss.data[0])


def load_data_mnist(batch_size=config.batch_size, test=False):
    train_path = '/data/lisa/data/mnist/mnist-python/train.pkl'
    if not os.path.isfile(train_path):
        raise Exception('Incorrect path to data')

    else:
        
        train_data = pickle.load(open(train_path,'rb'), encoding='latin1')
        mins = np.min(train_data['data'],axis=1)
        maxs = np.max(train_data['data'],axis=1)
        train_data['data'] = (train_data['data']-mins[:,None])/(maxs[:,None]- mins[:,None])
        train_dataset = TensorDataset(torch.Tensor(train_data['data']), torch.IntTensor(train_data['labels']))
        trainloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        if (test):
            test_path = '/data/lisa/data/mnist/mnist-python/test.pkl'
            test_data = pickle.load(open(test_path,'rb'), encoding='latin1')
            mins = np.min(test_data['data'],axis=1)
            maxs = np.max(test_data['data'],axis=1)
            test_data['data'] = (test_data['data']-mins[:,None])/(maxs[:,None]- mins[:,None])
            test_dataset = TensorDataset(torch.Tensor(test_data['data']), torch.IntTensor(test_data['labels']))
            testloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
            return trainloader, testloader

        return trainloader, None

def load_data_celeba(batch_size=config.batch_size,max_files = 0,test_split=0.2):
    save_root = './data/'
    traindir = save_root+'train/'
    testdir = save_root+'test/'
    train_loader = DataLoader(
        datasets.ImageFolder(traindir,
        transforms.Compose([
            transforms.CenterCrop(140),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0,0,0], [1,1,1])
        ])),
        batch_size=config.batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        datasets.ImageFolder(testdir,
        transforms.Compose([
            transforms.CenterCrop(140),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0,0,0],[1,1,1])
        ])),
        batch_size=100,
        shuffle=False
    )

    return train_loader, test_loader

def create_celeba_datapaths(split=0.9):
    root = '/data/'
    data_path = root + 'lisa/data/celeba/'
    save_root = './data/'
    traindir = save_root+'train/'
    testdir = save_root+'test/'

    split = 0.9

    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    if not os.path.isdir(traindir):
        os.mkdir(traindir)
        os.mkdir(traindir+'unsup')
    if not os.path.isdir(testdir):
        os.mkdir(testdir)
        os.mkdir(testdir+'unsup')
    if not os.path.isdir(save_root + 'celebA'):
        os.mkdir(save_root + 'celebA')

    img_list = os.listdir(data_path + 'img_align_celeba/')
    max_images = int(len(img_list))
    for i in range(len(img_list)):
        if (i%1000 ==0 ):
            print('Prepared {} images'.format((i)))
        img = plt.imread(data_path + 'img_align_celeba/' + img_list[i])
        save_dir = [traindir, testdir][i//int(max_images*split)]
        plt.imsave(fname=save_dir + 'unsup/' + img_list[i], arr=img)     

    
    

def truncate_noise(noise):
    return (100*noise).round()/100


def save_images(recon_x,x,epoch,confs):
    save_path = './images/{}/{}/'.format(confs['dataset'],confs['type'])
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_image(x, '{}/epoch_{}_data.jpg'.format(save_path,epoch), nrow=6,padding=2)
    save_image(recon_x.data,'{}/epoch_{}_recon.jpg'.format(save_path,epoch), nrow=6,padding=2)

def save_model(name,model,epoch,confs):
    save_path = './models/{}/{}/'.format(confs['dataset'],confs['type'])
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(),"{}/{}_{}.pt".format(save_path,name,epoch))
    
def load_model(fname,model):
    loaded_state_dict = torch.load("/data/milatmp1/goyettky/IFT6135/Project/WAE/models/{}".format(fname))
    state_dict = model.state_dict()
    state_dict.update(loaded_state_dict)
    model.load_state_dict(loaded_state_dict)

def save_losses(data,confs,epoch,test_train):
    save_path = './losses/{}/{}/'.format(confs['dataset'],confs['type'])
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    fname = save_path + '{}_{}'.format(test_train,epoch)
    with open(fname+'.pkl','wb') as f:
        pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)