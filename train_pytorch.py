# 檔名: train_pytorch.py
"""
手寫數字識別模型訓練程式 (PyTorch 版本)
使用 PyTorch 建立卷積神經網路 (CNN) 模型
"""

# 載入必要的函式庫
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

# 為了讓實驗可重現，設定隨機種子
torch.manual_seed(42)
np.random.seed(42)

def select_device():
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
                choice_str = input(f"請輸入您想使用的 GPU 索引 (0-{num_gpus-1})，或直接按 Enter 使用 CPU: ")
                if not choice_str:
                    print("未選擇 GPU。訓練將在 CPU 上運行。")
                    return torch.device("cpu")

                choice = int(choice_str)
                if 0 <= choice < num_gpus:
                    device = torch.device(f"cuda:{choice}")
                    print(f"已選擇 GPU {choice} ({torch.cuda.get_device_name(choice)})。")
                    return device
                else:
                    print("無效的選擇，請輸入正確的 GPU 索引。")
            except ValueError:
                print("無效的輸入，請輸入一個數字。")
    else:
        print("未偵測到 GPU 設備。訓練將在 CPU 上運行。")
        return torch.device("cpu")

def get_data_loaders(batch_size=300):
    """
    載入 MNIST 資料集並建立訓練、驗證和測試的 DataLoader

    Args:
        batch_size (int): 每個批次的圖像數量

    Returns:
        tuple: 包含訓練、驗證和測試的 DataLoader
    """
    # 定義訓練資料的轉換，包括資料增強
    # 數值是根據原 Keras 版本中的設定進行調整
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), shear=0.3, scale=(0.92, 1.08)),
        transforms.ToTensor(), # 將圖像轉為 Tensor，並將像素值標準化到 [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 資料集的標準化參數
    ])

    # 驗證和測試資料只需標準化，不需增強
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
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
    for i in range(9):
        plt.subplot(3, 3, i+1)
        img, label = vis_dataset[i]
        plt.imshow(img.squeeze(), cmap="gray") # squeeze() 移除通道維度
        plt.title(f"Class {label}")
    plt.tight_layout()
    plt.show()

    return train_loader, val_loader, test_loader

class CNN(nn.Module):
    """
    卷積神經網路模型架構
    繼承自 torch.nn.Module
    """
    def __init__(self):
        super(CNN, self).__init__()
        # PyTorch 的 Conv2d 輸入格式為 (N, C, H, W)
        # 第一個卷積層和池化層
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二個卷積層和池化層
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全連接層
        # 圖像經過兩次 2x2 池化，尺寸從 28x28 變為 7x7
        # 展平後的特徵數量為 32 (通道數) * 7 * 7 = 1568
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10) # 輸出層，10個類別

    def forward(self, x):
        """定義模型的前向傳播路徑"""
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # 輸出層不需手動加 softmax，因為 CrossEntropyLoss 會自動處理
        x = self.fc2(x)
        return x

def show_train_history(history, train_metric, val_metric):
    """
    繪製訓練歷史圖表 (準確率或損失)

    Args:
        history (dict): 包含訓練歷史記錄的字典
        train_metric (str): 訓練指標的鍵 (例如 'train_acc', 'train_loss')
        val_metric (str): 驗證指標的鍵 (例如 'val_acc', 'val_loss')
    """
    plt.figure(figsize=(10, 6))
    metric_name = train_metric.split('_')[1].capitalize()
    plt.plot(history[train_metric], label=f'Train {metric_name}')
    plt.plot(history[val_metric], label=f'Validation {metric_name}')
    plt.title("Training History")
    plt.ylabel(metric_name)
    plt.xlabel('Train cycles (Epoch)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def train_and_validate(model, device, train_loader, val_loader, optimizer, criterion, scheduler, epochs=30):
    """
    執行模型的訓練和驗證

    Args:
        model (nn.Module): 要訓練的模型
        device (torch.device): 計算設備 (CPU or GPU)
        train_loader (DataLoader): 訓練資料載入器
        val_loader (DataLoader): 驗證資料載入器
        optimizer: 優化器
        criterion: 損失函式
        scheduler: 學習率排程器
        epochs (int): 訓練週期數

    Returns:
        dict: 包含每個週期的訓練和驗證損失與準確率的歷史記錄
    """
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    model.to(device) # 將模型移至指定設備

    for epoch in range(epochs):
        # --- 訓練模式 ---
        model.train()
        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度歸零
            optimizer.zero_grad()

            # 前向傳播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向傳播與優化
            loss.backward()
            optimizer.step()

            # 統計損失和準確率
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # --- 驗證模式 ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad(): # 在驗證時不計算梯度
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        # 調整學習率
        scheduler.step(val_epoch_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        
    return history

def evaluate_model(model, device, test_loader, criterion):
    """
    在測試集上評估模型性能

    Args:
        model (nn.Module): 訓練好的模型
        device (torch.device): 計算設備
        test_loader (DataLoader): 測試資料載入器
        criterion: 損失函式

    Returns:
        tuple: (測試集準確率, 所有真實標籤, 所有預測標籤)
    """
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f"\n測試集準確率: {accuracy * 100:.2f}%")
    
    return accuracy, np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    """
    (此函式與原版相同)
    繪製混淆矩陣
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Real label', fontsize=12)
    plt.xlabel('Predict label', fontsize=12)
    plt.grid(False)
    plt.show()

def visualize_predictions(test_loader, y_test, predicted_classes):
    """
    視覺化預測結果 (錯誤和正確的範例)
    """
    # 取得未經標準化的測試圖像
    raw_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    # 找出錯誤預測的樣本
    incorrect_indices = np.where(y_test != predicted_classes)[0]
    plt.figure(figsize=(10, 10))
    plt.suptitle("Incorrect Predictions", fontsize=16)
    for i, idx in enumerate(incorrect_indices[:9]):
        plt.subplot(3, 3, i+1)
        img, _ = raw_test_dataset[idx]
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"Real: {y_test[idx]} / Predict: {predicted_classes[idx]}")
        plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # 找出正確預測的樣本
    correct_indices = np.where(y_test == predicted_classes)[0]
    plt.figure(figsize=(10, 10))
    plt.suptitle("Correct Predictions", fontsize=16)
    for i, idx in enumerate(correct_indices[:9]):
        plt.subplot(3, 3, i+1)
        img, label = raw_test_dataset[idx]
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"Category: {label}")
        plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# 主程式執行流程
if __name__ == "__main__":
    # 1. 設定
    DEVICE = select_device()
    BATCH_SIZE = 300
    EPOCHS = 30
    LEARNING_RATE = 0.001
    MODEL_SAVE_PATH = "pytorch_cnn.pth"

    # 2. 載入並準備資料
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)

    # 3. 建立模型、損失函式和優化器
    model = CNN()
    print("\n模型架構:")
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 設定學習率排程器 (等同於 Keras 的 ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'max',            # 當監控指標停止上升時觸發
        factor=0.5,       # 學習率 new_lr = lr * factor
        patience=3,       # 3 個 epoch 沒改善就調整
        verbose=True
    )

    # 4. 訓練模型
    print("\n--- 開始訓練 ---")
    history = train_and_validate(model, DEVICE, train_loader, val_loader, optimizer, criterion, scheduler, epochs=EPOCHS)
    print("--- 訓練完成 ---\n")

    # 5. 評估模型
    show_train_history(history, 'train_acc', 'val_acc')
    show_train_history(history, 'train_loss', 'val_loss')
    
    accuracy, y_test, y_pred = evaluate_model(model, DEVICE, test_loader, criterion)
    
    # 6. 分析結果 (混淆矩陣)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=range(10), title='Confusion Matrix')
    plot_confusion_matrix(cm, classes=range(10), normalize=True, title='Normalized Confusion Matrix')

    # 7. 視覺化預測
    visualize_predictions(test_loader, y_test, y_pred)
    
    # 8. 儲存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"模型狀態已儲存至 {MODEL_SAVE_PATH}")