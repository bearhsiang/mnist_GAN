import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 64
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.G = Generator(latent_size=self.latent_size).to(self.device)
        self.D = Discriminator().to(self.device)

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
            batch_size=32, 
            shuffle=True)
        
        epochs = 10
        for epoch in range(epochs):
            for train_D_iter in range(1):
                self.train_D(dataloader)

            for train_G_iter in range(1):
                self.train_G()

    def train_D(self, dataloader):
        self.D.train()
        self.G.eval()
        optimizer = optim.Adam(self.D.parameters(), lr=1e-3)
        total = 0
        bar = tqdm(dataloader)
        for data, labels in bar:
            optimizer.zero_grad()
            x_real = data.to(self.device)
            x_fake = self.generate_fake(data.shape[0])[0]
            predict_real = self.D(x_real)
            predict_fake = self.D(x_fake)
            loss = torch.log(predict_real).mean() + torch.log(1-predict_fake).mean()
            bar.set_postfix(
                loss = '{:.03f}'.format(loss.item()),
            )
            loss.backward()
            total += data.shape[0]


    def train_G(self):
        pass
    def generate_latent(self, N):
        return torch.randn((N, self.latent_size)).to(self.device)

    def generate_fake(self, N):
        x_fake = self.G(self.generate_latent(N))
        y_fake = torch.zeros(N, dtype=torch.long)
        return x_fake, y_fake

class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.linear = nn.Linear(latent_size, 128*7*7)
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 128, 4, 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 1, 7),
            nn.Sigmoid()
        )

    def forward(self, latent):

        base_img = self.linear(latent)
        base_img = F.leaky_relu(base_img)
        base_img = base_img.view(-1, 128, 7, 7)
        img = self.cnn(base_img)
        
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ## in, out, kernel_size
            nn.Conv2d(1, 64, 3, stride=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Conv2d(64, 64, 3, stride=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Flatten(),
            nn.Linear(2304, 1),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.model(input)
        
