#MNIST
conf_mnist={}
conf_mnist['latentd'] = 8
conf_mnist['dataset'] = 'MNIST'
conf_mnist['CUDA'] = False
conf_mnist['lr'] = 10e-3
conf_mnist['B1'] = 0.5
conf_mnist['B2'] = 0.999

celeba_latentd = 64

batch_size = 100

SAVEFREQ = 10
REPORTFREQ = 5
NUMEPOCHS = 1