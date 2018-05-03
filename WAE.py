import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

class WAE_GAN(nn.Module):
    def __init__(self,confs):
        super(WAE_GAN,self).__init__()
        self.confs = confs
        if confs['dataset'] == 'MNIST':
            self.encoder = Encoder_MNIST()
            self.decoder = Decoder_MNIST()
            self.discriminator = Discriminator(confs)
        elif confs['dataset'] == 'celeba':
            self.encoder = Encoder_Celeba()
            self.decoder = Decoder_Celeba()
            self.discriminator = Discriminator(confs)
        self.myparameters = nn.ParameterList(list(self.encoder.parameters()) + list(self.decoder.parameters()))

        if self.confs['CUDA']:
            self.mse = nn.MSELoss().cuda()
        else:
            self.mse = nn.MSELoss()

        self.discrim_loss = nn.BCEWithLogitsLoss()
            
    def encode(self,x):
        return self.encoder.forward(x)

    def decode(self,z):
        return self.decoder.forward(z)

    def reparameterize(self,mu,logvar,n=config.batch_size):
        if self.training:
            std = torch.exp(0.5*logvar)
            if self.confs['CUDA']:
                eps = torch.autograd.Variable(torch.cuda.FloatTensor(logvar.shape).normal_())
            else:
                eps = torch.autograd.Variable(torch.FloatTensor(logvar.shape).normal_())

            return eps.float().mul(std).add_(mu)
        else:
            return mu

    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu,logvar)
        recon_x = self.decode(z)

        return recon_x, mu, logvar

    def loss(self,recon_x,x,z,z_tilde):

        mse_loss = torch.sum(self.mse(recon_x,x))
        
        p_preds = self.discriminator.forward(z)
        q_preds = self.discriminator.forward(z_tilde)
        
        penalty = self.discrim_loss(q_preds,torch.ones_like(q_preds))
        #for discriminator
        loss_q = self.discrim_loss(q_preds,torch.zeros_like(q_preds))
        loss_p = self.discrim_loss(p_preds,torch.ones_like(p_preds))

        d_loss = self.confs['lambda']*(loss_q + loss_p)/config.batch_size
        enc_dec_loss = (mse_loss + self.confs['lambda']*penalty)/config.batch_size
        return enc_dec_loss, d_loss
            

    def pretain_loss(self,mu,logvar):
        sample_noise = torch.autograd.Variable(torch.cuda.FloatTensor(logvar.shape).normal_())
        sample_q = self.reparameterize(mu,logvar)
        mean_pz = torch.mean(sample_noise,dim=0)
        mean_qz = torch.mean(sample_q,dim=0)
        mean_loss = torch.mean(torch.square(mean_pz-mean_qz))
        cov_pz = torch.matmul((sample_noise - mean_pz).t(), sample_noise-mean_pz)
        cov_pz /= config.batch_size -1
        cov_qz = torch.matmul((sample_q-mean_qz).t(),(sample_q-mean_qz))
        conv_qz /= config.batch_size -1
        cov_loss = torch.mean(torch.square(cov_pq-cov_qz))
        return mean_loss + cov_loss


class WAE_MMD(nn.Module):
    def __init__(self,confs):
        super(WAE_MMD,self).__init__()
        self.confs = confs
        if confs['dataset'] == 'MNIST':
            self.encoder = Encoder_MNIST()
            self.decoder = Decoder_MNIST()
        elif confs['dataset'] == 'celeba':
            self.encoder = Encoder_Celeba()
            self.decoder = Decoder_Celeba()
        self.myparameters = nn.ParameterList(list(self.encoder.parameters()) + list(self.decoder.parameters()))

        if self.confs['CUDA']:
            self.mse = nn.MSELoss(size_average=False).cuda()
        else:
            self.mse = nn.MSELoss(size_average=False)

    def encode(self,x):
        return self.encoder.forward(x)

    def decode(self,z):
        return self.decoder.forward(z)

    def reparameterize(self,mu,logvar,n=config.batch_size):
        

        if self.training:
            std = torch.exp(0.5*logvar)
            std = torch.clamp(std,-50,50)
            if self.confs['CUDA']:
                eps = torch.autograd.Variable(torch.cuda.FloatTensor(logvar.shape).normal_())
            else:
                eps = torch.autograd.Variable(torch.FloatTensor(logvar.shape).normal_())

            return eps.float().mul(std).add_(mu)
        else:
            return mu

    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu,logvar)
        recon_x = self.decode(z)

        return recon_x, mu, logvar

    def kernel(self,z_0,z_1):
        C = 2.0*self.confs['latentd']*(self.confs['sig_z']**2)
        return C/(C+torch.mean(z_0-z_1)**2)

    def loss(self,recon_x,x,z,z_tilde):

        mse_loss = self.mse(recon_x,x)
        
        qz_norm = torch.sum(z_tilde**2,dim=1,keepdim=True)
        pz_norm = torch.sum(z**2,dim=1,keepdim=True)
        qzqz_dot = torch.matmul(z_tilde,z_tilde.t())
        pzpz_dot = torch.matmul(z,z.t())
        qzpz_dot = torch.matmul(z_tilde, z.t())

        qz_dist = qz_norm + qz_norm.transpose(1,0) - 2.0 * qzqz_dot
        pz_dist = pz_norm + pz_norm.transpose(1,0) - 2.0*pzpz_dot
        qzpz_dist = qz_norm + pz_norm.transpose(1,0) -2.0 *qzpz_dot

        C_init = 2.0*self.confs['latentd']*(self.confs['sig_z']**2)

        mmd_loss = 0
        for scale in [0.1,0.2,0.5,1.0,2.0,5.0,10.0]:
            C = C_init * scale
            res1 = C/ (C+ qz_dist) + C/(C+pz_dist)
            res1 = torch.matmul(res1, torch.autograd.Variable(1.0-torch.eye(config.batch_size).cuda()))
            res1 = torch.sum(res1)/(config.batch_size*(config.batch_size-1))
            res2 = C/(C+qzpz_dist)
            res2 = 2* torch.sum(res2)/(config.batch_size**2)

            mmd_loss += res1 - res2

        return mse_loss + self.confs['lambda']*mmd_loss

class Discriminator(nn.Module):
    def __init__(self,confs):
        super(Discriminator,self).__init__()
        self.confs = confs
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=confs['latentd'],out_features=512),
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
        if self.confs['n_trick']:
            adder = torch.sum(torch.square(x))/2/(self.confs['sig_z']**2) - 0.5*torch.log(2*torch.pi) - 0.5 * self.confs['latentd']*torch.log(self.confs['sig_z']**2)
            return F.sigmoid(self.layer4(self.layer3(self.layer2(self.layer1(x))))) + adder

        else:
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

    