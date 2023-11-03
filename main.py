import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import animation, rc

import cv2
import math
import random

def torch_fix_seed(seed=1):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


torch_fix_seed()


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.lr1 = nn.Linear(3*64*64, 300)
        self.lr2 = nn.Linear(300, 100)
        self.lr_ave = nn.Linear(100, z_dim)   #average
        self.lr_dev = nn.Linear(100, z_dim)   #log(sigma^2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        x = self.relu(x)
        ave = self.lr_ave(x)    #average
        log_dev = self.lr_dev(x)    #log(sigma^2)

        ep = torch.randn_like(ave)   #平均0分散1の正規分布に従い生成されるz_dim次元の乱数
        z = ave + torch.exp(log_dev / 2) * ep   #再パラメータ化トリック
        return z, ave, log_dev

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.lr1 = nn.Linear(z_dim, 100)
        self.lr2 = nn.Linear(100, 300)
        self.lr3 = nn.Linear(300, 3*64*64)
        self.relu = nn.ReLU()
    
    def forward(self, z):
        x = self.lr1(z)
        x = self.relu(x)
        x = self.lr2(x)
        x = self.relu(x)
        x = self.lr3(x)
        x = torch.sigmoid(x)
        return x

class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
    
    def forward(self, x):
        z, ave, log_dev = self.encoder(x)
        x = self.decoder(z)
        return x, z, ave, log_dev
    

def criterion(predict, target, ave, log_dev):
    bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
    loss = bce_loss + kl_loss
    return loss



class JarTestDataset(Dataset):
    def __init__(self, is_train=True, valid_id=0):
        super(JarTestDataset, self).__init__()

        self.is_train = is_train
        self.mean = 0
        self.std = 1

        dataset_dir_path = os.path.join('..', 'vp_dataset', 'jar_tests')
        dataset_path = os.path.join(dataset_dir_path, 'dataset_mini.npy')
        self.dataset = np.load(dataset_path)

        # 水質データの読み込み
        dataset_dir_path = os.path.join('..', 'vp_dataset', 'jar_tests')
        wq_data_path = os.path.join(dataset_dir_path, 'water_quality.csv')
        df_water = pd.read_csv(wq_data_path)

        # 水質データを最終濁度についてソートし，idのnumpy配列を得る
        idx_array = df_water.sort_values('fin_turbidity')['id'].values

        # idx配列をsplit_num行の2次元配列にする
        # あまりの部分は -1 で埋める
        split_num = 5
        rows = split_num
        cols = math.ceil(len(idx_array)/split_num)
        total_elements = rows * cols
        remainder_array = -1*np.ones(total_elements - len(idx_array))
        idx_array = np.concatenate((idx_array, remainder_array)).astype(np.int32)
        idx_mat = idx_array.reshape(cols, rows).T
        
        T_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 24, 28, 32, 44, 51, 69, 89, 134, 239, 314, 599]
        self.dataset = self.dataset[:, T_idx, ...]
        print(self.dataset.shape)


        # train test に分割
        if is_train:
            # 訓練データはidに対応する行以外を抜き出す
            video_id_array = np.delete(idx_mat, valid_id, 0).flatten()
        else:
            # 検証データはid に対応する行を抜き出す
            video_id_array = idx_mat[valid_id]
            
        # あまりの -1 は削除
        video_id_array = video_id_array[video_id_array != -1]
        self.dataset = self.dataset[video_id_array]
        self.dataset = self.dataset.astype(np.float32)


        N, T, C, H, W = self.dataset.shape
        self.dataset = self.dataset.reshape((N*T, C, H, W))
        print(self.dataset.shape)


    def __getitem__(self, idx):
        data = self.dataset[idx]
        output = torch.from_numpy(data / 255.0).contiguous().float().to(torch.device('cuda'))
        return output

    def __len__(self):
        return len(self.dataset)
    

def main():
    # # データセットの読み込み
    # dataset_dir_path = os.path.join('..', 'vp_dataset', 'jar_tests')
    # dataset_path = os.path.join(dataset_dir_path, 'dataset_mini.npy')
    # dataset = np.load(dataset_path)

    # print(dataset.shape)

    BATCH_SIZE = 100
    train_data = JarTestDataset(is_train=True)
    val_data = JarTestDataset(is_train=False)

    train_loader = DataLoader(dataset=train_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=0)

    val_loader = DataLoader(dataset=val_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=0)

    print("train data size: ",len(train_data))
    print("train iteration number: ",len(train_data)//BATCH_SIZE)
    print("val data size: ",len(val_data))
    print("val iteration number: ",len(val_data)//BATCH_SIZE)

    images = next(iter(train_loader))
    print("images_size: " ,images.size())   #images_size: torch.Size([100, 1, 28, 28])

    # image_numpy = images.detach().numpy().copy()
    # plt.imshow(image_numpy[0,0,:,:], cmap='gray')

    z_dim = 2
    num_epochs = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = VAE(z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)

    history = {"train_loss": [], "val_loss": [], "ave": [], "log_dev": [], "z": [], "labels":[]}

    for epoch in range(num_epochs):
        model.train()
        for i, x in enumerate(train_loader):
            input = x.to(device).view(-1, 3*64*64).to(torch.float32)
            output, z, ave, log_dev = model(input)

            history["ave"].append(ave)
            history["log_dev"].append(log_dev)
            history["z"].append(z)
            loss = criterion(output, input, ave, log_dev)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 50 == 0:
                print(f'Epoch: {epoch+1}, loss: {loss: 0.4f}')
            history["train_loss"].append(loss)

        model.eval()
        with torch.no_grad():
            for i, x in enumerate(val_loader):
                input = x.to(device).view(-1, 3*64*64).to(torch.float32)
                output, z, ave, log_dev = model(input)

                loss = criterion(output, input, ave, log_dev)
                history["val_loss"].append(loss)
                
            print(f'Epoch: {epoch+1}, val_loss: {loss: 0.4f}')
        
        scheduler.step()
    
    train_loss_tensor = torch.stack(history["train_loss"])
    train_loss_np = train_loss_tensor.to('cpu').detach().numpy().copy()
    plt.plot(train_loss_np)
    val_loss_tensor = torch.stack(history["val_loss"])
    val_loss_np = val_loss_tensor.to('cpu').detach().numpy().copy()
    plt.plot(val_loss_np)
    plt.show()

    ave_tensor = torch.stack(history["ave"])
    log_var_tensor = torch.stack(history["log_dev"])
    z_tensor = torch.stack(history["z"])
    print(ave_tensor.size())   #torch.Size([9600, 100, 2])
    print(log_var_tensor.size())   #torch.Size([9600, 100, 2])
    print(z_tensor.size())   #torch.Size([9600, 100, 2])

    ave_np = ave_tensor.to('cpu').detach().numpy().copy()
    log_var_np = log_var_tensor.to('cpu').detach().numpy().copy()
    z_np = z_tensor.to('cpu').detach().numpy().copy()
    print(ave_np.shape)   #(9600, 100, 2)
    print(log_var_np.shape)   #(9600, 100, 2)
    print(z_np.shape)   #(9600, 100, 2)

    # model.encoder
    # print()

    fig = plt.figure()
    num = 7
    for n in range(num*num):
        i = n // num
        i = float(0.5 - (i / 0.8))
        j = n % num
        j = float(0.5 - (j / 0.8))
        print(i, j)

        img = model.decoder(torch.Tensor([i, j]).to(torch.float32).to(device))
        img = img.to('cpu').detach().numpy().copy()
        img = img.reshape((3, 64, 64))
        img = img.transpose(1, 2, 0)

        fig.add_subplot(num, num, n+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


    # for i in range(10):
        
    #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     plt.show()

    map_keyword = "tab10"
    cmap = plt.get_cmap(map_keyword)

    # batch_num =10
    # plt.figure(figsize=[10,10])
    # for label in range(10):
    #     x = z_np[:batch_num,:,0][labels_np[:batch_num,:] == label]
    #     y = z_np[:batch_num,:,1][labels_np[:batch_num,:] == label]
    #     plt.scatter(x, y, color=cmap(label/9), label=label, s=15)
    #     plt.annotate(label, xy=(np.mean(x),np.mean(y)),size=20,color="black")
    # plt.legend(loc="upper left")
    # plt.show()

    

    # batch_num = 9580
    # plt.figure(figsize=[10,10])
    # for label in range(10):
    #     x = z_np[batch_num:,:,0][labels_np[batch_num:,:] == label]
    #     y = z_np[batch_num:,:,1][labels_np[batch_num:,:] == label]
    #     plt.scatter(x, y, color=cmap(label/9), label=label, s=15)
    #     plt.annotate(label, xy=(np.mean(x),np.mean(y)),size=20,color="black")
    # plt.legend(loc="upper left")
    # plt.show()


if __name__ == '__main__':
    main()
