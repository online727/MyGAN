import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import time
from operator import itemgetter
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim import Adam

from model import *


class config:
    # seed
    seed = 26
    use_seed = True

    # models setting
    models_setting = {
        "from_old_model": False,
        "G_type": "L",
        "D_type": "C",
    }

    # training setting
    training_setting = {
        "num_epochs": 10,
        "batch_size": 64,
        "lr": 0.0003,
        "loss": nn.BCELoss(),
        "img_seed_dim": 100,  # image dimension
        "D_train_times": 2,
        "G_train_times": 1,
        "D_train_label_exchange": 0.05,  # exchange real and fake labels
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    def __init__(self):
        self.seed_all()
        self.name = f"G_{self.models_setting['G_type']}_D_{self.models_setting['D_type']}"
        G_model_path = f"model/{self.name}_G_model.pth"
        D_model_path = f"model/{self.name}_D_model.pth"
        img_output_path = f"output_images/{self.name}"

        self.pathes = {
            "data_path": 'data',
            "G_model_path": G_model_path,
            "D_model_path": D_model_path,
            "img_output_path": img_output_path,
            "save_name": self.name
        }

        if not os.path.exists(img_output_path):
            os.makedirs(img_output_path)

        print(f"Using device: {self.training_setting['device']}")

    def seed_all(self):
        if self.use_seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __getattr__(self, item):
        if not isinstance(item, list):
            item = [item]
        lst = []
        for i in item:
            if i in self.models_setting:
                lst.append(self.models_setting[i])
            elif i in self.training_setting:
                lst.append(self.training_setting[i])
            elif i in self.pathes:
                lst.append(self.pathes[i])
            else:
                raise AttributeError(f"'config' object has no attribute '{item}'")
        return lst if len(lst) > 1 else lst[0]

    @property
    def models(self):
        return self.models_setting

    @property
    def training(self):
        return self.training_setting

    @property
    def path(self):
        return self.pathes


class get_my_dataset(Dataset):
    def __init__(
            self, 
            data_path, 
            batch_size=64,
            features_transform=None, 
            target_transform=None
            ):
        
        self.data_path = data_path
        self.features_transform = features_transform
        self.target_transform = target_transform
        self.batch_size = batch_size
        self.get_data_loader()

    def get_data(self):
        # len = 60000, each element is a tuple of (image, label), image's shape (1, 28, 28)
        self.minist_train = torchvision.datasets.MNIST(
            root=self.data_path,
            train=True,
            transform=self.features_transform,
            target_transform=self.target_transform,
            download=False
        )

        # len = 10000, each element is a tuple of (image, label), image's shape (1, 28, 28)
        self.minist_test = torchvision.datasets.MNIST(
            root=self.data_path,
            train=False,
            transform=self.features_transform,
            target_transform=self.target_transform,
            download=False
        )
    
    def get_data_loader(self):
        self.get_data()

        # len = 60000, each element is a tuple of (image, label), image's shape (1, 28, 28)
        self.minist_train_loader = DataLoader(
            dataset=self.minist_train,
            batch_size=self.batch_size,
            shuffle=True
        )

        # len = 10000, each element is a tuple of (image, label), image's shape (1, 28, 28)
        self.minist_test_loader = DataLoader(
            dataset=self.minist_test,
            batch_size=self.batch_size,
            shuffle=False
        )
        return self.minist_train_loader, self.minist_test_loader

    @property
    def train_data_loader(self):
        return self.minist_train_loader

    @property
    def test_data_loader(self):
        return self.minist_test_loader


def get_model(from_old_model, G_model_path, D_model_path, device, G_type, D_type):
    G_model = get_Generator(
        from_old_model=from_old_model, model_path=G_model_path, device=device, G_type=G_type
        )
    
    D_model = get_Discriminator(
        from_old_model=from_old_model, model_path=D_model_path, device=device, D_type=D_type
        )
    return G_model, D_model


def train(
        models, optimizers, train_set, config
        ):

    G_model, D_model = models['G'], models['D']
    G_optimizer, D_optimizer = optimizers['G'], optimizers['D']
    G_model_path, D_model_path, img_path = config.__getattr__(["G_model_path", "D_model_path", "img_output_path"])
    epochs, batch_size, lr, criterion, device = \
        config.__getattr__(
            ["num_epochs", "batch_size", "lr", "loss", "device"]
        )
    img_seed_dim, D_train_times, G_train_times, D_train_label_exchange = \
        config.__getattr__(
            ["img_seed_dim", "D_train_times", "G_train_times", "D_train_label_exchange"]
        )
    D_loss_lst = []
    G_loss_lst = []

    train_start = time.time()
    for epoch in range(epochs):
        print(f'start epoch: {epoch + 1} -----------------------')

        batch_num = len(train_set)
        D_loss_sum = 0.0
        G_loss_sum = 0.0
        count = 0

        for index, (images, _) in enumerate(train_set):
            count += 1
            img_num = images.size(0)
            real_images = images.to(device)

            real_labels = Variable(1 - (torch.rand(img_num) / 10).to(device))
            fake_labels = Variable(torch.zeros(img_num).to(device))
            exchange_labels = False

            for _ in range(D_train_times):
                fake_images = G_model(torch.randn(img_num, img_seed_dim).to(device))

                if random.uniform(0, 1) < D_train_label_exchange:
                    real_labels, fake_labels = fake_labels, real_labels
                    exchange_labels = True

                real_output = D_model(real_images)
                D_loss_real = criterion(real_output, real_labels)
                fake_output = D_model(fake_images)
                D_loss_fake = criterion(fake_output, fake_labels)

                D_loss = D_loss_real + D_loss_fake
                D_loss_sum += D_loss.item()

                D_optimizer.zero_grad()
                D_loss.backward()
                D_optimizer.step()

                if exchange_labels:
                    real_labels, fake_labels = fake_labels, real_labels

            for _ in range(G_train_times):
                fake_images = G_model(torch.randn(img_num, img_seed_dim).to(device))
                fake_output = D_model(fake_images)

                G_loss = criterion(fake_output, real_labels)
                G_loss_sum += G_loss.item()

                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()

            if (index + 1) % 200 == 0:
                print(f'[epoch: {epoch + 1}/{epochs} batch: {index + 1}/{batch_num}] '
                      f'D_loss: {D_loss.item():.6f}, G_loss: {G_loss.item():.6f}')
        
        torch.save(G_model.state_dict(), G_model_path)
        torch.save(D_model.state_dict(), D_model_path)

        with torch.no_grad():
            fake_images = G_model(torch.randn(batch_size, img_seed_dim).to(device)).cpu()

            fake_images = (fake_images + 1) * 0.5
            fake_images = fake_images.clamp(0, 1)

            fake_images = fake_images.view(-1, 1, 28, 28)
            save_image(fake_images, f'{img_path}/epoch_{epoch + 1}.png')

            g_loss = G_loss_sum / count / G_train_times
            d_loss = D_loss_sum / count / D_train_times
            G_loss_lst.append(g_loss)
            D_loss_lst.append(d_loss)

            print(f'Epoch {epoch + 1}/{epochs}, D_loss: {d_loss:.6f}, G_loss: {g_loss:.6f}, '
                  f'Using Time: {(time.time() - train_start) / 60:.4f} min')

    print(f'Training Finished, Using Time: {(time.time() - train_start) / 60:.4f} min')
    return G_loss_lst, D_loss_lst


if __name__ == '__main__':
    my_config = config()

    features_transform = transforms.Compose([
        # 将像素值从 [0, 255] 转换到 [0, 1]
        transforms.ToTensor(),
        # 将像素值从 [0, 1] 转换到 [-1, 1], 将输入的图像按照通道进行标准化
        # MINIST 数据集是灰度图像，只有一个通道，所以 mean 和 std 都是一个数
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    target_transform = transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.float32))

    my_data = get_my_dataset(
        data_path=my_config.data_path,
        batch_size=my_config.batch_size,
        features_transform=features_transform,
        target_transform=target_transform
    )
    train_dataloader = my_data.train_data_loader
    test_dataloader = my_data.test_data_loader

    G_model, D_model = get_model(
        my_config.from_old_model,
        my_config.G_model_path,
        my_config.D_model_path,
        my_config.device,
        my_config.G_type,
        my_config.D_type
    )

    G_optimizer = Adam(
        G_model.parameters(), lr=my_config.lr
        )
    D_optimizer = Adam(
        D_model.parameters(), lr=my_config.lr
        )

    models = {
        'G': G_model,
        'D': D_model
    }
    optimizers = {
        'G': G_optimizer,
        'D': D_optimizer
    }

    G_loss_lst, D_loss_lst = train(
        models=models, optimizers=optimizers, train_set=train_dataloader, config=my_config
        )

    plt.figure()
    plt.plot(G_loss_lst, label='G_loss')
    plt.plot(D_loss_lst, label='D_loss')
    plt.legend()
    plt.suptitle(f'{my_config.save_name} G and D loss')
    plt.savefig(f'{my_config.save_name}_loss.png')

    df = pd.DataFrame({
        'G_loss': G_loss_lst,
        'D_loss': D_loss_lst
    })
    df.to_csv(f'{my_config.save_name}_loss.csv', index=False)