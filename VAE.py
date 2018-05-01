import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

class VAE(nn.Module):
    def __init__(self,confs):
        super(VAE,self).__init__()
        self.confs = confs
        if confs['dataset'] == 'MNIST':
            self.encoder = Encoder_MNIST()
            self.decoder = Decoder_MNIST()
        elif confs['dataset'] == 'celeba':
            self.encoder = Encoder_Celeba()
            self.decoder = Decoder_Celeba()
        self.myparameters = nn.ParameterList(list(self.encoder.parameters()) + list(self.decoder.parameters()))
        if self.confs['dataset'] == 'MNIST':
            if self.confs['CUDA']:
                self.bce = nn.BCELoss(size_average=False).cuda()
            else:
                self.bce = nn.BCELoss(size_average=False)
        elif self.confs['dataset'] == 'celeba':
            if self.confs['CUDA']:
                self.mse = nn.MSELoss(size_average=False).cuda()
            else:
                self.mse = nn.MSELoss(size_average=False)
            
    def encode(self,x):
        return self.encoder.forward(x)

    def decode(self,z):
        return self.decoder.forward(z)

    def reparameterize(self,mu,logvar,n=config.batch_size):
        
        if self.confs['CUDA']:
            eps = torch.autograd.Variable(torch.cuda.FloatTensor(logvar.shape).normal_())
        else:
            eps = torch.autograd.Variable(torch.FloatTensor(logvar.shape).normal_())

        return eps.float().mul(logvar.mul(0.5).exp()).add_(mu)

    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu,logvar)
        recon_x = self.decode(z)

        return recon_x, mu, logvar

    def loss(self,recon_x,x,mu,logvar):
        if self.confs['dataset'] == 'MNIST':
            bce_loss = self.bce(recon_x,x)

            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            return bce_loss + KLD, bce_loss, KLD
        elif self.confs['dataset'] == 'celeba':
            mse_loss = self.mse(recon_x,x)

            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            return mse_loss + KLD, mse_loss, KLD
            

    def pretrain_loss(self,mu,logvar):
        pass
        

class Encoder_MNIST(nn.Module):
    def __init__(self):
        super(Encoder_MNIST,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=2,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2,padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2,padding=2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2,padding=0),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU())
        self.fc_layer_mean = nn.Linear(in_features=1024*1*1, out_features=8)
        self.fc_layer_logvar = nn.Linear(in_features=1024*1*1,out_features=8)

    def forward(self,x):

        h4 = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        h4 = h4.view(-1,1024*1*1)
        mu = self.fc_layer_mean(h4)
        logvar = self.fc_layer_logvar(h4)

        return mu, logvar

class Decoder_MNIST(nn.Module):
    def __init__(self):
        super(Decoder_MNIST,self).__init__()
        self.layer1 = nn.Linear(in_features=8, out_features=7*7*1024)
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,out_channels=512,stride=2,padding=1,kernel_size=4),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=256,stride=2,padding=1, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer4 = nn.ConvTranspose2d(in_channels=256,out_channels=1, stride=1, padding=1,kernel_size=3)
        

    def forward(self,x):
        h1 = self.layer1(x)
        h1 = h1.view(-1,1024,7,7)
        return F.sigmoid(self.layer4(self.layer3(self.layer2(h1))))


class Decoder_Celeba(nn.Module):
    def __init__(self):
        super(Decoder_Celeba,self).__init__()
        self.layer1 = nn.Linear(in_features=64,out_features=8*8*1024)
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=5,stride=2,padding=2, output_padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,out_channels=3,kernel_size=5,stride=1,padding=2)
        )
    def forward(self,x):
        h1 = self.layer1(x)
        h1 = h1.view(-1,1024,8,8)
        output = self.layer4(self.layer3(self.layer2(h1)))
        return output.add_(torch.autograd.Variable(torch.cuda.FloatTensor(output.shape).normal_().mul_(0.3)))

class Encoder_Celeba(nn.Module):
    def __init__(self):
        super(Encoder_Celeba,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=128,kernel_size=5,padding=2,stride=2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=5,padding=2,stride=2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=5,padding=2,stride=2),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5,padding=2,stride=2),
            nn.ReLU(),
        )
        self.layer5_mu = nn.Linear(in_features=1024*4*4,out_features=64)
        self.layer5_logvar = nn.Linear(in_features=1024*4*4,out_features=64)

    def forward(self,x):
        h4 = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        h4 = h4.view(-1,1024*4*4)
        return self.layer5_mu(h4), self.layer5_logvar(h4)
        



        