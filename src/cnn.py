
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from util import *
import data_decoder as dc


n_epochs = 8
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)
m_transform = torchvision.transforms.Normalize((0.1307,), (0.3081,))

def load_data():
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("../", train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                m_transform])),
    batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("../", train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                m_transform])),
    batch_size=batch_size_test, shuffle=True)
    examples = enumerate(test_loader)
    batch_idx, (exampledata, exampletargets) = next(examples)
    #print(exampledata.shape)
    return train_loader, test_loader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 5
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50) 
        self.fc2 = nn.Linear(50, 10)
        self.transform = torchvision.transforms.Normalize((0.1307,), (0.3081,))


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320) #320
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


    def predict(self, img):
        print(img)
        ft = m_transform(torch.FloatTensor(img).unsqueeze(0))
        ft = ft.unsqueeze(0)
        #print(ft)
        pred = self.forward(ft)
        pred = torch.argmax(pred, dim=1)
        #print(pred)
        return pred.item()


def load_cnn_model():
    model = Net()
    checkpoint = torch.load(MODEL_DIR + CNN_MODEL_NAME)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def main():
    
    train_loader, test_loader = load_data()

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)


    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                torch.save(network.state_dict(), f'../models/{CNN_MODEL_NAME}')
                torch.save(optimizer.state_dict(), '../models/optimizer.pth')


    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()


if __name__ == "__main__":
    main()
