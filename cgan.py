import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import save_img

eps = 1e-9
class GAN(nn.Module):
    def __init__(self, device, batch_size, class_num, class_emb):
        super().__init__()
        self.latent_size = 64
        self.device = device
        self.batch_size = batch_size
        self.class_num = class_num
        self.class_emb = class_emb
        self.G = Generator(self.latent_size, class_num, class_emb)
        self.D = Discriminator(class_num, class_emb)

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
                y_real = labels.to(self.device)
                x_fake, y_fake = self.generate_fake(self.batch_size)
                predict_real = self.D(x_real, y_real)
                predict_fake = self.D(x_fake, y_fake)

                wrong_labels = self.generate_labels(self.batch_size)
                predict_wrong = self.D(x_real, wrong_labels)
                predict_wrong = predict_wrong[wrong_labels != y_real]

                loss_D_real = - torch.log(predict_real).mean()
                loss_D_fake = - torch.log(torch.clamp(1-predict_fake, min=eps)).mean()
                loss_D_wrong = - torch.log(torch.clamp(1-predict_wrong, min=eps)).mean()

                loss_D = loss_D_real + loss_D_fake + loss_D_wrong
                loss_D.backward()
                optim_D.step()
                
                acc_D = (torch.sum(predict_real>0.5).item()+torch.sum(predict_fake < 0.5).item())/self.batch_size/2
                acc_D_W = torch.sum(predict_wrong < 0.5).item()/predict_wrong.shape[0]

                ### train_G
                self.G.train()
                self.D.eval()
                optim_G.zero_grad()
                x_fake, y_fake = self.generate_fake(self.batch_size)
                predict_fake = self.D(x_fake, y_fake)
                loss_G = - torch.log(predict_fake).mean()
                loss_G.backward()
                optim_G.step()
                acc_G = torch.sum(predict_fake < 0.5).item()/predict_fake.shape[0]

                bar.set_postfix(
                    loss_D = '{:.03f}'.format(loss_D),
                    acc_D = '{:.03f}'.format(acc_D),
                    loss_G = '{:.03f}'.format(loss_G),
                    acc_G = '{:.03f}'.format(acc_G),
                    acc_D_W = '{:.03f}'.format(acc_D_W)
                )            
                
            # img = self.generate_fake(1)[0][0].detach().cpu()
            # save_img(img, f'output/{epoch}.jpg')


    def generate_latent(self, N):
        return torch.randn((N, self.latent_size)).to(self.device)

    def generate_labels(self, N):
        return torch.randint(self.class_num, (N, )).to(self.device)

    def generate_fake(self, N):
        y_fake = self.generate_labels(N)
        latents = self.generate_latent(N)
        x_fake = self.G(latents, y_fake)
        return x_fake, y_fake

class Generator(nn.Module):
    def __init__(self, latent_size, class_num, class_emb):
        super().__init__()
        self.latent_linear = nn.Linear(latent_size, 128*7*7)
        self.embedding = nn.Embedding(class_num, class_emb)
        self.emb_linear = nn.Linear(class_emb, 7*7)
        self.cnn = nn.Sequential(
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(),

            nn.ConvTranspose2d(129, 64, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, latents, labels):

        base_imgs = self.latent_linear(latents)
        base_imgs = F.leaky_relu(base_imgs)
        base_imgs = base_imgs.view(-1, 128, 7, 7)

        class_imgs = self.embedding(labels)
        class_imgs = self.emb_linear(class_imgs)
        class_imgs = class_imgs.view(-1, 1, 7, 7)

        imgs = torch.cat((base_imgs, class_imgs), dim=1)
        imgs = self.cnn(imgs)
        
        return imgs

class Discriminator(nn.Module):
    def __init__(self, class_num, class_emb):
        super().__init__()
        self.embedding = nn.Embedding(class_num, class_emb)
        self.emb_linear = nn.Linear(class_emb, 28*28)
        self.model = nn.Sequential(
            ## in, out, kernel_size
            nn.Conv2d(2, 64, 3, stride=2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.4),

            nn.Conv2d(64, 64, 3, stride=2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.4),

            nn.Flatten(),
            nn.Linear(2304, 1),
        )
    def forward(self, imgs, labels):
        emb = self.embedding(labels)
        class_imgs = self.emb_linear(emb).view(-1, 1, 28, 28)
        imgs = torch.cat((imgs, class_imgs), dim=1)
        output = self.model(imgs)
        return torch.sigmoid(output)
