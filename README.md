# WassersteinAutoEncoder

Implementation of Wasserstein Auto Encoders in Pytorch as discussed in: 
https://arxiv.org/pdf/1711.01558.pdf

To train a network:
python run.py --exp={mnist_vae,mnist_wae_mmd,mnist_wae_gan,celeba_vae,celeba_wae_mmd,celeba_wae_gan}

Acknowledgements to://
https://github.com/tolstikhin/wae//
https://github.com/mseitzer/pytorch-fid
