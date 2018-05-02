#MNIST-VAE
conf_mnist_vae={}
conf_mnist_vae['type'] = 'VAE'
conf_mnist_vae['latentd'] = 8
conf_mnist_vae['dataset'] = 'MNIST'
conf_mnist_vae['CUDA'] = False
conf_mnist_vae['lr'] = 10e-3
conf_mnist_vae['B1'] = 0.5
conf_mnist_vae['B2'] = 0.999
conf_mnist_vae['milestones1'] = [30]
conf_mnist_vae['milestones2'] = [50]
conf_mnist_vae['NUMEPOCHS'] = 100
conf_mnist_vae['noise'] = True
conf_mnist_vae['loss'] = 'vae'

#Celeba-VAE
conf_celeba_vae = {}
conf_celeba_vae['type'] = 'VAE'
conf_celeba_vae['latentd'] = 64
conf_celeba_vae['dataset'] = 'celeba'
conf_celeba_vae['CUDA'] = True
conf_celeba_vae['lr'] = 10e-4
conf_celeba_vae['B1'] = 0.5
conf_celeba_vae['B2'] = 0.999
conf_celeba_vae['milestones1'] = [30]
conf_celeba_vae['milestones2'] = [50]
conf_celeba_vae['NUMEPOCHS'] = 68
conf_celeba_vae['noise'] = False
conf_celeba_vae['loss'] = 'vae'

#MNIST-WAE-GAN
conf_mnist_wae_gan = {}
conf_mnist_wae_gan['type'] = 'WAEGAN'
conf_mnist_wae_gan['latentd'] = 8
conf_mnist_wae_gan['dataset'] = 'MNIST'
conf_mnist_wae_gan['CUDA'] = True
conf_mnist_wae_gan['lr'] = 10e-3
conf_mnist_wae_gan['B1'] = 0.5
conf_mnist_wae_gan['B2'] = 0.999
conf_mnist_wae_gan['lr_disc'] = 5*10e-4
conf_mnist_wae_gan['B1_disc'] = 0.5
conf_mnist_wae_gan['B2_disc'] = 0.999
conf_mnist_wae_gan['milestones1'] = [30]
conf_mnist_wae_gan['milestones2'] = [50]
conf_mnist_wae_gan['NUMEPOCHS'] = 100
conf_mnist_wae_gan['noise'] = True
conf_mnist_wae_gan['loss'] = 'wae-gan'
conf_mnist_wae_gan['lambda'] = 10
conf_mnist_wae_gan['n_trick'] = False
conf_mnist_wae_gan['sig_z'] = 1

#Celeba-WAE-GAN
conf_celeba_wae_gan = {}
conf_celeba_wae_gan['type'] = 'WAEGAN'
conf_celeba_wae_gan['latentd'] = 64
conf_celeba_wae_gan['dataset'] = 'celeba'
conf_celeba_wae_gan['CUDA'] = True
conf_celeba_wae_gan['lr'] = 3e-4
conf_celeba_wae_gan['B1'] = 0.5
conf_celeba_wae_gan['B2'] = 0.999
conf_celeba_wae_gan['lr_disc'] = 10e-3
conf_celeba_wae_gan['B1_disc'] = 0.5
conf_celeba_wae_gan['B2_disc'] = 0.999
conf_celeba_wae_gan['milestones1'] = [30]
conf_celeba_wae_gan['milestones2'] = [50]
conf_celeba_wae_gan['NUMEPOCHS'] = 55
conf_celeba_wae_gan['noise'] = False
conf_celeba_wae_gan['loss'] = 'wae-gan'
conf_celeba_wae_gan['lambda'] = 1
conf_celeba_wae_gan['n_trick'] = True
conf_celeba_wae_gan['sig_z'] = 1

#MNIST-WAE-MMD
conf_mnist_wae_mmd = {}
conf_mnist_wae_mmd['type'] = 'WAEMMD'
conf_mnist_wae_mmd['latentd'] = 8
conf_mnist_wae_mmd['dataset'] = 'MNIST'
conf_mnist_wae_mmd['CUDA'] = True
conf_mnist_wae_mmd['lr'] = 10e-3
conf_mnist_wae_mmd['B1'] = 0.5
conf_mnist_wae_mmd['B2'] = 0.999
conf_mnist_wae_mmd['milestones1'] = [30]
conf_mnist_wae_mmd['milestones2'] = [50]
conf_mnist_wae_mmd['NUMEPOCHS'] = 100
conf_mnist_wae_mmd['noise'] = True
conf_mnist_wae_mmd['loss'] = 'wae-mmd'
conf_mnist_wae_gan['sig_z'] = 1

#MNIST-WAE-MMD
conf_celeba_wae_mmd = {}
conf_celeba_wae_mmd['type'] = 'WAEMMD'
conf_celeba_wae_mmd['latentd'] = 8
conf_celeba_wae_mmd['dataset'] = 'celeba'
conf_celeba_wae_mmd['CUDA'] = True
conf_celeba_wae_mmd['lr'] = 10e-3
conf_celeba_wae_mmd['B1'] = 0.5
conf_celeba_wae_mmd['B2'] = 0.999
conf_celeba_wae_mmd['milestones1'] = [30]
conf_celeba_wae_mmd['milestones2'] = [50]
conf_celeba_wae_mmd['NUMEPOCHS'] = 55
conf_celeba_wae_mmd['noise'] = False
conf_celeba_wae_mmd['loss'] = 'wae-mmd'
conf_celeba_wae_gan['sig_z'] = 1

batch_size = 100

SAVEFREQ = 1
REPORTFREQ = 1