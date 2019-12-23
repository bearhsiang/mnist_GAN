import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.DataLoader as DataLoader
import torchvision
class GAN(nn.Module):
    def __init__(self):
        self.super().__init__()
        self.G = Generator()
        self.D = Discriminator()
    def train(self, args):
        mnist_dataset = torchvision.datasets.MNIST()
        dataloader = DataLoader(mnist_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=True,)
        epochs = 10
        for epoch in range(epochs):
            for train_D_iter in range(train_D_iters):
                self.train_D()
            for train_G_iter in range(train_G_iters):
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
        self.super().__init__()
        self.model = nn.Sequential()
        pass
    def forward(self, latent):
        return self.model(latent)

class Discriminator(nn.Module):
    def __init__(self):
        self.super().__init__()
        self.model = nn.Sequential()
        pass
    def forward(self, input):
        return self.model(input)
        
