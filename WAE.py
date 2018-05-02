import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

class WAE_GAN(nn.Module):
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

        self.discrim_loss = nn.BCEWithLogitsLoss()
            
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

    def loss(self,recon_x,x,z,z_tilde):

        mse_loss = torch.sum(self.mse(recon_x,x))
        mse_loss = mse_loss/config.batch_size

        p_preds = self.discriminator.forward(z)
        q_preds = self.discriminator.forward(z_tilde)

        penalty = self.discrim_loss(logits_q,torch.ones_like(logits_q))
        
        loss_q = self.discrim_loss(q_preds,torch.zeros_like(q_preds))
        loss_p = self.discrim_loss(p_preds,torch.ones_like(p_preds))

        d_loss = self.confs['lambda']*(loss_q + loss_p)/config.batch_size
        enc_dec_loss = mse_loss + self.confs['lambda']*penalty
        return enc_dec_loss, d_loss
            

    def pret_loss(self,mu,logvar):
        pass

class Discriminator_MNIST(nn.Module):
    def __init__(self):
        super(Discriminator_MNIST,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=8,out_features=512),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=512,out_features=512),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=1)
        )

    def forward(self,x):
        return F.sigmoid(self.layer4(self.layer3(self.layer2(self.layer1(x)))))


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

    