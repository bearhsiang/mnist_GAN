import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from cgan import GAN



if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 128
    
    mnist_dataset = datasets.MNIST(
        '/hdd/torchvision/',
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    )

    model = GAN(device = device,
            batch_size=batch_size,
            class_num=10,
            class_emb=64,
        ).to(device)

    dataloader = DataLoader(mnist_dataset,
        batch_size=batch_size, 
        shuffle=False,
        )
    
    model.train(dataloader)

