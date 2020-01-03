import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import save_img

eps = 1e-9
class GAN(nn.Module):
    def __init__(self, device, batch_size):
        super().__init__()
        self.latent_size = 64
        self.device = device
        self.batch_size = batch_size
        self.G = Generator(latent_size=self.latent_size)
        self.D = Discriminator()

    def train(self, dataloader, epochs = 100):

        

        optim_D = optim.Adam(self.D.parameters(), lr=1e-4)
        optim_G = optim.Adam(self.G.parameters(), lr=1e-4)

        for epoch in range(epochs):

            bar = tqdm(dataloader)

            for batch, (data, labels) in enumerate(bar):

                
                ### train_D
                self.D.train()
                self.G.eval()
                optim_D.zero_grad()
                x_real = data.to(self.device)
                x_fake = self.generate_fake(self.batch_size)[0]
                predict_real = self.D(x_real)
                predict_fake = self.D(x_fake)
                loss_D_real = - torch.log(predict_real).mean()
                loss_D_fake = - torch.log(torch.clamp(1-predict_fake, min=eps)).mean()
                loss_D = loss_D_real + loss_D_fake
                loss_D.backward()
                optim_D.step()
                
                acc_D = (torch.sum(predict_real>0.5).item()+torch.sum(predict_fake < 0.5).item())/self.batch_size/2

                ### train_G
                self.G.train()
                self.D.eval()
                optim_G.zero_grad()
                x_fake = self.generate_fake(self.batch_size)[0]
                predict_fake = self.D(x_fake)
                loss_G = - torch.log(predict_fake).mean()
                loss_G.backward()
                optim_G.step()
                acc_G = torch.sum(predict_fake < 0.5).item()/predict_fake.shape[0]

                bar.set_postfix(
                    loss_D = '{:.03f}'.format(loss_D),
                    acc_D = '{:.03f}'.format(acc_D),
                    loss_G = '{:.03f}'.format(loss_G),
                    acc_G = '{:.03f}'.format(acc_G)
                )            
                
            img = self.generate_fake(1)[0][0].detach().cpu()
            save_img(img, f'output/{epoch}.jpg')


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
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
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
            # nn.Dropout(0.4),

            nn.Conv2d(64, 64, 3, stride=2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.4),

            nn.Flatten(),
            nn.Linear(2304, 1),
        )
    def forward(self, input):
        output = self.model(input)
        return torch.sigmoid(output)
