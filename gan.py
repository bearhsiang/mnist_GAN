import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator()
    def train(self):
        mnist_dataset = torchvision.datasets.MNIST(
            '/hdd/torchvision/',
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081, )),
                ])
            )

        dataloader = DataLoader(mnist_dataset,
            batch_size=8, 
            shuffle=True)
        
        epochs = 10
        for epoch in range(epochs):
            for train_D_iter in range(1):
                for data, labels in dataloader:
                    print(data.shape, labels.shape)
                    self.train_D()

            for train_G_iter in range(1):
                self.train_G()
    def train_D(self):
        pass
    def train_G(self):
        pass
    def generate_latent(self):
        return torch.randn((self.batch_size, self.latent))

    def generate_fake(self):
        x_fake = self.G(self.generate_latent())
        y_fake = torch.zeros(self.batch_size, dtype=torch.long)
        return x_fake, y_fake

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential()
        pass
    def forward(self, latent):
        return self.model(latent)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
        )
        pass
    def forward(self, input):
        return self.model(input)
        
