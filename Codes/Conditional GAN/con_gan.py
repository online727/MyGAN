import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import sklearn
import os

figs_save_path = 'figs/'
os.makedirs(figs_save_path, exist_ok=True)

# Define the Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, num_channel, num_gen_fea, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, num_gen_fea * 8),
            nn.BatchNorm1d(num_gen_fea * 8),
            nn.ReLU(True),
            nn.Linear(num_gen_fea * 8, num_gen_fea * 16),
            nn.BatchNorm1d(num_gen_fea * 16),
            nn.ReLU(True),
            nn.Linear(num_gen_fea * 16, num_channel * 28 * 28),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        input = torch.cat((noise, label_embedding), -1)
        output = self.model(input)
        output = output.view(output.size(0), 1, 28, 28)
        return output


# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_channel, num_dis_fea, num_classes):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(num_channel * 28 * 28 + num_classes, num_dis_fea * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_dis_fea * 8, num_dis_fea * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_dis_fea * 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        img = img.view(img.size(0), -1)
        label_embedding = self.label_emb(labels)
        d_in = torch.cat((img, label_embedding), -1)
        validity = self.model(d_in)
        return validity


# Hyperparameters
BATCH_SIZE = 64
LR = 0.0002
NUM_EPOCHS = 50

NOISE_DIM = 100     # Noise dimension
NUM_CLASSES = 10    # Number of classes
NUM_CHANNEL = 1     # Number of channels, minist is grayscale
NUM_GEN_FEA = 64    # Number of features in generator
NUM_DIS_FEA = 64    # Number of features in discriminator


# load data and preprocess
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='../Original GAN/data', train=True, transform=transform, download=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# define the device and initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(NOISE_DIM, NUM_CHANNEL, NUM_GEN_FEA, NUM_CLASSES).to(device)
discriminator = Discriminator(NUM_CHANNEL, NUM_DIS_FEA, NUM_CLASSES).to(device)


# define the loss function and optimizer
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))


# Training
progress_bar = tqdm(range(NUM_EPOCHS))
for epoch in progress_bar:
    progress_bar.set_description(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
    g_loss_all = 0.0
    d_loss_all = 0.0

    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.size(0)

        real_imgs = imgs.to(device)
        labels = labels.to(device)

        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)


        # train the generator

        optimizer_G.zero_grad()

        z = torch.randn(batch_size, NOISE_DIM).to(device)
        gen_labels = torch.randint(0, NUM_CLASSES, (batch_size,)).to(device)

        gen_imgs = generator(z, gen_labels)

        g_loss = criterion(discriminator(gen_imgs, gen_labels), valid)
        g_loss_all += g_loss.item()

        g_loss.backward()
        optimizer_G.step()

        
        # train the discriminator

        optimizer_D.zero_grad()

        real_loss = criterion(discriminator(real_imgs, labels), valid)
        fake_loss = criterion(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = real_loss + fake_loss
        d_loss_all += d_loss.item()

        d_loss.backward()
        optimizer_D.step()

    progress_bar.set_postfix(g_loss=g_loss_all/(i+1), d_loss=d_loss_all/(i+1))
    
    # save the generated images every epoch
    with torch.no_grad():
        # 生成噪声
        test_z = torch.randn(60, NOISE_DIM).to(device)
        # 创建标签，每个数字重复6次
        test_labels = torch.tensor([i for i in range(10) for _ in range(6)]).to(device)
        # 生成图像
        test_imgs = generator(test_z, test_labels).cpu()
        # 创建图像网格，每行10个图像，共6行
        grid = torchvision.utils.make_grid(test_imgs, nrow=6, normalize=True)
        # 绘制并保存图像
        plt.figure(figsize=(3, 6))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(figs_save_path + f'{epoch+1}.png')
        plt.close()

    # save the model every 10 epochs
    if (epoch+1) % 10 == 0:
        torch.save(generator.state_dict(), f'generator_{epoch+1}.pth')


def generate_digit(digit, num_images):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, NOISE_DIM).to(device)
        labels = torch.full((num_images,), digit, dtype=torch.long).to(device)
        imgs = generator(z, labels).cpu()
        for i in range(num_images):
            plt.imshow(imgs[i].squeeze(), cmap='gray')
            plt.axis('off')
            plt.savefig(figs_save_path + f'{digit}_{i}.png')
            plt.close()


for i in range(10):
    generate_digit(i, 3)

torch.save(generator.state_dict(), 'generator_final.pth')