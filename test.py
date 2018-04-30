from utils import load_data_mnist

trainloader, testloader = load_data_mnist(100)

for batch_ind, (data,_) in enumerate(trainloader):
    print(data.shape)
