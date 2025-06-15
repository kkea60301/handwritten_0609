import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import os
import random # 導入 random 模組

torch.manual_seed(42)
np.random.seed(42)
random.seed(42) # 設定 random 模組的種子

def select_device():  #定義函式 - 配置 PyTorch 以使用 GPU 或 CPU
    """
    配置 PyTorch 以使用 GPU 或 CPU

    列出所有可用的 GPU 設備，讓使用者選擇要使用的 GPU。
    如果沒有 GPU 或使用者選擇不使用，則退回使用 CPU。

    Returns:
        torch.device: 代表所選計算設備的物件 (例如 "cuda:0" 或 "cpu")
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("偵測到以下 GPU 設備：")
        for i in range(num_gpus):
            print(f"  {i}: {torch.cuda.get_device_name(i)}")

        while True:
            try:
                choice_str = input(f"請輸入您想使用的 GPU Index (0-{num_gpus-1})，或直接按 Enter 使用 CPU: ")
                if not choice_str:
                    print("未選擇 GPU。訓練將在 CPU 上運行。")
                    return torch.device("cpu")

                choice = int(choice_str)
                if 0 <= choice < num_gpus:
                    device = torch.device(f"cuda:{choice}")
                    print(f"已選擇 GPU {choice} ({torch.cuda.get_device_name(choice)})。")
                    return device
                else:
                    print("無效的選擇，請輸入正確的 GPU Index。")
            except ValueError:
                print("無效的輸入，請輸入一個數字。")
    else:
        print("未偵測到 GPU 設備。訓練將在 CPU 上運行。")
        return torch.device("cpu")

def get_data_loaders(batch_size=300): #接受一個參數batch_size（批次大小），預設值是 300。是指每次訓練時同時處理多少張圖片。
    """
    載入 MNIST 資料集並建立訓練、驗證和測試的 DataLoader

    Args:
        batch_size (int): 每個批次的圖像數量

    Returns:
        tuple: 包含訓練、驗證和測試的 DataLoader
    """
    # 定義訓練資料的轉換，包括資料增強
    # 數值是根據原 Keras 版本中的設定進行調整
    # transforms.Compose將多種transforms變換組合在一起
    train_transform = transforms.Compose([  
        transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), shear=0.3, scale=(0.92, 1.08)), #隨機對圖片進行旋轉、平移、剪切和縮放，讓模型看到更多變化的圖片
        transforms.ToTensor(), # 將PIL(Python Imaging Library)圖像轉換成 PyTorch 可以處理的張量格式，並將像素值從 0-255 轉換到 0-1
        transforms.Normalize((0.1307,), (0.3081,)) # 使用 MNIST 資料集的統計數據進行標準化(pixel_value - mean) / std, NIST的MEAN=0.1307和STD=0.3081
    ])

    # 驗證和測試資料只需標準化，不需增強
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))# 使用 MNIST 資料集的統計數據進行標準化(pixel_value - mean) / std, NIST的MEAN=0.1307和STD=0.3081
    ])

    # 下載並載入完整的訓練資料集
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    
    # 為了視覺化，我們也載入一份未經轉換的資料
    vis_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    # 將訓練資料集分割為訓練集和驗證集 (80/20)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # 下載並載入測試資料集
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    # 建立 DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # 視覺化部分訓練樣本 (使用未轉換的資料)
    plt.figure(figsize=(4, 4))
    # 隨機選擇 9 張圖片的索引
    random_indices = random.sample(range(len(vis_dataset)), 9)
    for i, idx in enumerate(random_indices):
        plt.subplot(3, 3, i+1)
        img, label = vis_dataset[idx] # 使用隨機索引
        plt.imshow(img.squeeze(), cmap="gray") # squeeze() 移除通道維度
        plt.title(f"Class {label}")
    plt.tight_layout()
    plt.show()

    return train_loader, val_loader, test_loader

# 執行 GPU 檢查
if __name__ == "__main__":
    # 1. 設定
    DEVICE = select_device()
    BATCH_SIZE = 300
    EPOCHS = 30
    LEARNING_RATE = 0.001
    MODEL_SAVE_PATH = "pytorch_cnn.pth"

    # 2. 載入並準備資料
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)
    device = select_device()
    print(f"使用的計算設備: {device}") 


