import torch
import torch.nn as nn
from utils import save_images, load_model
from torchvision.utils import save_image
import config
import WAE
import VAE
import os

def traverse_latent(model,num_images=10):
    z1 = torch.autograd.Variable(torch.normal(torch.zeros(num_images,model.confs['latentd']),std=1)).cuda() 
    z2 = torch.autograd.Variable(torch.normal(torch.zeros(num_images,model.confs['latentd']),std=1)).cuda() 

    alphas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    results = None

    for alpha in alphas:
        z = alpha*z1 + (1-alpha*z2)
        #z = torch.autograd.Variable(z).cuda()
        recon_x = model.decode(z)
        if type(results) == type(None):
           results = recon_x.data
        else:
            results = torch.cat((results,recon_x.data),dim=0)
    path = './latent_trav/{}/{}/'.format(model.confs['dataset'],model.confs['type'])
    if not os.path.isdir(path):
        os.makedirs(path,)
    save_image(results, '{}/trav_full.jpg'.format(path,alpha), nrow=num_images,padding=2)

