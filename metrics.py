from fid_score import calculate_frechet_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_model, load_data_celeba, load_data_mnist
from WAE import WAE_GAN, WAE_MMD
from VAE import VAE
import config

def calc_average_fid_and_blur(model,dloader):
    fid=0
    blur=0
    model.eval()
    for batch_ind, (data,_) in enumerate(dloader):
        if model.confs['CUDA']:
            data = data.cuda()
        data = torch.autograd.Variable(data.view(1,1,28,28))
        #data.unsqueeze(0)
        print(data.shape)
        recon_x, _, _ = model.forward(data)

        blur += calc_blur(recon_x).data[0]
        recon_x = recon_x.view(1,recon_x.shape[1]*recon_x.shape[2]*recon_x.shape[3])
        #recon_x.unsqueeze()


        data = data.view(1,data.shape[1]*data.shape[2]*data.shape[3])
        recon_acts_mean = np.mean(recon_x.data.cpu().numpy(),axis=0)
        recon_acts_sig = np.cov(recon_x.data.cpu().numpy(),rowvar=False)


        data_acts_mean = np.mean(data.data.cpu().numpy(),axis=0)
        data_acts_sig = np.cov(data.data.cpu().numpy(),rowvar=False)
        #print(recon_acts_sig)
        # Source: https://github.com/mseitzer/pytorch-fid
        fid += calculate_frechet_distance(data_acts_mean,data_acts_sig,recon_acts_mean,recon_acts_sig,eps=1e-8)
        #print(fid)
        print(blur)

    print('Fid: {}'.format(fid/len(dloader.dataset)))
    print('Blur: {}'.format(blur/len(dloader.dataset)))


                                
def calc_blur(image):
    
    #https://github.com/tolstikhin/wae/blob/master/wae.py
    if image.size(1) == 3:
        image = torch.mean(X, 1, keepdim=True)

    laplace_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplace_filter = laplace_filter.reshape([1, 1, 3, 3])
    laplace_filter = torch.autograd.Variable(torch.from_numpy(laplace_filter).float())

    
    laplace_filter = lap_filter.cuda()

    # valid padding (i.e., no padding)
    conv = F.conv2d(X, laplace_filter, padding=0, stride=1)

    # smoothness is the variance of the convolved image
    var = torch.var(conv)
    return(var)