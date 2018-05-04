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
            self.mse = nn.MSELoss(size_average=False).cuda()
        else:
            self.mse = nn.MSELoss(size_average=False)

        self.discrim_loss = nn.CrossEntropyLoss().cuda()
            
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
        
        penalty = -torch.mean(torch.log(q_preds+1e-8))#self.discrim_loss(q_preds,torch.ones_like(q_preds))
        #for discriminator
        #loss_q = #self.discrim_loss(q_preds,torch.zeros_like(q_preds))
        #loss_p = #self.discrim_loss(p_preds,torch.ones_like(p_preds))
        
        d_loss = -self.confs['lambda']*torch.mean(torch.log(p_preds+1e-8) + torch.log(1-q_preds+1e-8))
        enc_dec_loss = (mse_loss + self.confs['lambda']*penalty)
        return enc_dec_loss, d_loss, mse_loss,penalty
            

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

    #def loss(self,recon_x,x,z,z_tilde):
    #    #print(torch.max(z_tilde).data[0])
    #    mse_loss = self.mse(recon_x,x)#

    #    C_init = 2.0*self.confs['latentd']*(self.confs['sig_z']**2)
        #print(z_tilde_tile1.shape)

    #    k_pp = C_init/(C_init + torch.sum((z.unsqueeze(0)-z.unsqueeze(1))**2,dim=2))
        #print(k_pp.shape)
    #    k_qq = C_init/(C_init + torch.sum((z_tilde.unsqueeze(0)-z_tilde.unsqueeze(1))**2,dim=2))
    #    k_pq = C_init/(C_init + torch.sum((z_tilde.unsqueeze(0)-z.unsqueeze(1))**2,dim=2))

    #    mmd_loss = (torch.sum(k_pp) - torch.sum(torch.diag(k_pp)) + torch.sum(k_qq) - torch.sum(torch.diag(k_qq)) - 2* (torch.sum(k_pq) - torch.sum(torch.diag(k_pq))))/(config.batch_size*(config.batch_size-1))   
        #print(mmd_loss.data[0])
    #    return mse_loss + self.confs['lambda']*mmd_loss
    
    def loss(self,recon_x,x,z,z_tilde):

        mse_loss = self.mse(recon_x,x)/recon_x.shape[0]
        
        qz_norm = torch.sum(z_tilde**2,dim=1,keepdim=True)
        pz_norm = torch.sum(z**2,dim=1,keepdim=True)
        qzqz_dot = torch.matmul(z_tilde,z_tilde.t())
        pzpz_dot = torch.matmul(z,z.t())
        qzpz_dot = torch.matmul(z_tilde, z.t())
        n = z_tilde.shape[0]
        qz_dist = qz_norm + qz_norm.transpose(1,0) - 2.0 * qzqz_dot
        pz_dist = pz_norm + pz_norm.transpose(1,0) - 2.0*pzpz_dot
        qzpz_dist = qz_norm + pz_norm.transpose(1,0) -2.0 *qzpz_dot
        C_init = 2.0*self.confs['latentd']*(self.confs['sig_z']**2)

        mmd_loss = 0
        for scale in [0.1,0.2,0.5,1.0,2.0,5.0,10.0]:
            C = C_init * scale
            res1 = C/ (C+ qz_dist) + C/(C+pz_dist)
            res1 = torch.matmul(res1, torch.autograd.Variable(1.0-torch.eye(n).cuda()))
            res1 = torch.sum(res1)/(n*(n-1))
            res2 = C/(C+qzpz_dist)
            res2 = 2* torch.sum(res2)/(n**2)

            mmd_loss += res1 - res2

        return mse_loss + self.confs['lambda']*mmd_loss, mse_loss,self.confs['lambda']*mmd_loss 

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
            adder = torch.sum(x**2,dim=1)/2/(self.confs['sig_z']**2) - torch.autograd.Variable(0.5*torch.log(2*torch.FloatTensor([np.pi])).cuda()) - torch.autograd.Variable(0.5 * self.confs['latentd']*torch.log(torch.FloatTensor(self.confs['sig_z']**2)).cuda())

            return F.sigmoid(self.layer4(self.layer3(self.layer2(self.layer1(x))))+adder)

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


class Decoder_Celeba(nn.Module):
    def __init__(self):
        super(Decoder_Celeba,self).__init__()
        self.layer1 = nn.Linear(in_features=64,out_features=8*8*1024)
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=5,stride=2,padding=2, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,out_channels=3,kernel_size=5,stride=1,padding=2)
        )
    def forward(self,x):
        h1 = self.layer1(x)
        h1 = h1.view(-1,1024,8,8)
        return self.layer4(self.layer3(self.layer2(h1)))

class Encoder_Celeba(nn.Module):
    def __init__(self):
        super(Encoder_Celeba,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=128,kernel_size=5,padding=2,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=5,padding=2,stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=5,padding=2,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5,padding=2,stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.layer5_mu = nn.Linear(in_features=1024*4*4,out_features=64)
        self.layer5_logvar = nn.Linear(in_features=1024*4*4,out_features=64)

    def forward(self,x):
        h4 = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        h4 = h4.view(-1,1024*4*4)
        return self.layer5_mu(h4), self.layer5_logvar(h4)