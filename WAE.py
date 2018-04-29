import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=2,padding=15),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2,padding=15),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2,padding=15),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2,padding=15),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU())
        self.fc_layer_mean = nn.Linear(in_features=1024*28*28, out_features=8)
        self.fc_layer_logvar = nn.Linear(in_features=1024*28*28,out_features=8)

    def forward(self,x):

        h4 = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        h4 = h4.view(-1,1024*28*28)
        mu = self.fc_layer_mean(h4)
        logvar = self.fc_layer_logvar(h4)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
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

        self.layer4 = nn.ConvTranspose2d(in_channels=256,out_channels=1, stride=1, padding=1,kernel_size=4)
        

    def forward(self,x):
        h1 = self.layer1(x)
        h1 = h1.view(-1,1024,7,7)
        return F.sigmoid(self.layer4(self.layer3(self.layer2(h1))))




x=Encoder()
                