#MNIST
conf_mnist_vae={}
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

celeba_latentd = 64

batch_size = 100

SAVEFREQ = 10
REPORTFREQ = 5
NUMEPOCHS = 1