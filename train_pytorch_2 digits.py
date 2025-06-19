# 檔名: train_pytorch_2 digits.py
"""
手寫雙位數數字識別模型訓練程式 (PyTorch 版本)
使用 PyTorch 建立卷積神經網路 (CNN) 模型，辨識 00-99 的雙位數數字。
"""

# 載入必要的函式庫，PyTorch 核心模組
import torch #PyTorch 的主要模組，提供張量運算和 GPU 計算功能
import torch.nn as nn #神經網路模組，用來建立模型架構
import torch.optim as optim #優化器模組，用來更新模型權重，包含各種訓練演算法（如 SGD、Adam 等）
import torch.nn.functional as F #提供各種神經網路函式，提供各種神經網路函式的功能版本

#以下這些模組專門處理資料
from torch.utils.data import DataLoader, random_split, Dataset # DataLoader 用來批次載入資料，random_split 用來隨機分割資料集, Dataset 用於自訂資料集
from torchvision import transforms # transforms 用來對圖像進行轉換和增強
from PIL import Image # 用於圖像處理

#科學計算與視覺化
import numpy as np #數值計算函式庫，提供高效的陣列運算
import matplotlib.pyplot as plt #視覺化函式庫，用來繪製圖表和圖像
from sklearn.metrics import confusion_matrix # 來自 scikit-learn，用於計算混淆矩陣評估模型性能
import itertools # itertools 模組提供高效的迭代器工具，這裡用於生成混淆矩陣的索引
import os # 用於檔案和目錄操作
import random # 導入 random 模組，用於隨機數生成

# 為了讓實驗可重現，設定隨機種子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42) # 也設定 Python 內建的 random 模組

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

class DoubleMNISTDataset(Dataset):
    """
    自訂資料集類別，用於載入雙位數 MNIST 圖像。
    圖像尺寸為 28x56，標籤為 0-99 的整數。
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 遍歷根目錄下的所有類別資料夾 (例如 '00', '01', ..., '99')
        for label_str in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label_str)
            if os.path.isdir(label_path):
                # 將字串標籤轉換為整數 (例如 '05' -> 5, '99' -> 99)
                label_int = int(label_str)
                for img_name in os.listdir(label_path):
                    if img_name.endswith('.png'):
                        self.image_paths.append(os.path.join(label_path, img_name))
                        self.labels.append(label_int)

        print(f"載入 {len(self.image_paths)} 個雙位數圖像從 {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L') # 轉換為灰度圖像

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(batch_size=300, data_root='./data/double_mnist'): # 調整預設資料路徑
    """
    載入雙位數 MNIST 資料集並建立訓練、驗證和測試的 DataLoader

    Args:
        batch_size (int): 每個批次的圖像數量
        data_root (str): 雙位數 MNIST 資料集的根目錄

    Returns:
        tuple: 包含訓練、驗證和測試的 DataLoader
    """
    # 雙位數 MNIST 資料集的標準化參數 (需要重新計算或使用近似值)
    # 這裡我們假設與單一 MNIST 相似，但更精確的做法是計算雙位數資料集的均值和標準差
    MEAN = (0.1307,) # 假設與單一 MNIST 相似
    STD = (0.3081,)  # 假設與單一 MNIST 相似

    # 定義訓練資料的轉換，包括資料增強
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), shear=0.3, scale=(0.92, 1.08)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # 驗證和測試資料只需標準化，不需增強
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # 載入雙位數訓練資料集
    full_train_dataset = DoubleMNISTDataset(root_dir=os.path.join(data_root, 'train'), transform=train_transform)
    
    # 將訓練資料集分割為訓練集和驗證集 (80/20)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # 載入雙位數測試資料集
    test_dataset = DoubleMNISTDataset(root_dir=os.path.join(data_root, 'test'), transform=test_transform)

    # 建立 DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

class CNN(nn.Module):
    """
    卷積神經網路模型架構，適用於雙位數 MNIST (28x56 圖像，100 個類別)
    繼承自 torch.nn.Module
    """
    def __init__(self):
        super(CNN, self).__init__()
        # PyTorch 的 Conv2d 輸入格式為 (N, C, H, W)
        # 輸入圖像尺寸為 28x56
        
        # 第一個卷積層和池化層
        # 輸入通道數為 1 (灰階圖片); 輸出 32 個特徵圖; 5x5 卷積核; 保持輸入輸出尺寸相同
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding='same') 
        # 2x2 池化窗口，步長為 2，將圖片尺寸縮小一半 (28x56 -> 14x28)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二個卷積層和池化層
        # 輸入通道數為 32; 輸出 32 個特徵圖; 5x5 卷積核; 保持輸入輸出尺寸相同
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding='same')
        # 2x2 池化窗口，步長為 2，將圖片尺寸縮小一半 (14x28 -> 7x14)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全連接層
        # 圖像經過兩次 2x2 池化，尺寸從 28x56 變為 7x14
        # 展平層 (flatten)：將多維的特徵圖轉換成一維向量
        # 輸入特徵數量為 32 * 7 * 14 = 3136
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 14, 256) # 調整輸入特徵數量
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 100) # 輸出層，100個類別 (00-99)

    def forward(self, x):
        """定義模型的前向傳播路徑"""
        # x 的形狀應為 (N, 1, 28, 56)，其中 N 是批次大小
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
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
    epochs_range = np.arange(1, len(history[train_metric]) + 1) # 建立從 1 開始的 epoch 範圍用於繪圖和標籤
    plt.plot(epochs_range, history[train_metric], label=f'Train {metric_name}')
    plt.plot(epochs_range, history[val_metric], label=f'Validation {metric_name}')
    plt.title("Training History")
    plt.ylabel(metric_name)
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='solid', alpha=0.4)
    # 確保 x 軸從 1 開始，y 軸從 0 開始
    plt.xlim(left=1) # 確保 x 軸從 1 開始
    plt.ylim(bottom=0) # 確保 y 軸從 0 開始
    plt.xticks(epochs_range) # 設定 x 軸刻度位置和標籤
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

    def _run_epoch(loader, model, criterion, optimizer=None, is_train=True):
        """
        輔助函式：執行一個訓練或驗證週期
        """
        if is_train:
            model.train()
        else:
            model.eval()

        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        
        with torch.set_grad_enabled(is_train):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)

                if is_train:
                    optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if is_train:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc

    for epoch in range(epochs):
        # 訓練模式
        train_loss, train_acc = _run_epoch(train_loader, model, criterion, optimizer, is_train=True)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 驗證模式
        val_loss, val_acc = _run_epoch(val_loader, model, criterion, is_train=False)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 調整學習率
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
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
    繪製混淆矩陣
    """
    plt.figure(figsize=(10, 10)) # 調整圖形大小
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=8) # 調整字體大小和旋轉
    plt.yticks(tick_marks, classes, fontsize=8)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=6) # 調整字體大小
    
    plt.tight_layout()
    plt.ylabel('Real label', fontsize=12)
    plt.xlabel('Predict label', fontsize=12)
    plt.grid(False)
    plt.show()

def visualize_predictions(model, device, test_dataset, y_test, predicted_classes):
    """
    視覺化預測結果 (錯誤和正確的範例)
    """
    # 找出錯誤預測的樣本
    incorrect_indices = np.where(y_test != predicted_classes)[0]
    
    # 確保模型處於評估模式
    model.eval()
    
    plt.figure(figsize=(12, 18)) # 調整圖形大小以適應 3x2 佈局 (圖像 + 柱狀圖)
    plt.suptitle("Incorrect Predictions with Probabilities", fontsize=16)
    
    # 隨機選擇錯誤預測的樣本進行顯示
    if len(incorrect_indices) > 0:
        display_indices = random.sample(list(incorrect_indices), min(len(incorrect_indices), 5))
    else:
        display_indices = []
        print("沒有錯誤預測的樣本可供視覺化。")
        
    with torch.no_grad():
        for i, idx in enumerate(display_indices):
            img, true_label = test_dataset[idx]
            input_img = img.unsqueeze(0).to(device)
            outputs = model(input_img)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy().flatten()
            
            # 繪製圖像
            ax1 = plt.subplot(len(display_indices), 2, 2*i + 1) # 圖像在左側
            ax1.imshow(img.squeeze(), cmap="gray")
            ax1.set_title(f"Real: {true_label:02d} / Predict: {predicted_classes[idx]:02d}", fontsize=10) # 格式化為兩位數
            ax1.axis('off')
            
            # 繪製機率柱狀圖
            ax2 = plt.subplot(len(display_indices), 2, 2*i + 2) # 柱狀圖在右側
            bars = ax2.bar(range(100), probabilities * 100, color='skyblue') # 100 個類別
            ax2.set_ylim(0, 100)
            ax2.set_xticks(np.arange(0, 100, 10)) # 調整 x 軸刻度，每 10 個顯示一個
            ax2.set_xlabel("Two-Digit Number")
            ax2.set_ylabel("Probability (%)")
            ax2.set_title("Prediction Probabilities", fontsize=10)
            
            # 在柱狀圖上顯示數值 (只顯示前幾個高機率的)
            top_k = 5 # 顯示前 5 個最高機率
            top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
            for k_idx in top_k_indices:
                yval = probabilities[k_idx] * 100
                ax2.text(k_idx, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=7, color='red')
            
    if len(display_indices) > 0:
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # 找出正確預測的樣本
    correct_indices = np.where(y_test == predicted_classes)[0]
    plt.figure(figsize=(10, 10))
    plt.suptitle("Correct Predictions", fontsize=16)
    for i, idx in enumerate(correct_indices[:9]):
        plt.subplot(3, 3, i+1)
        img, label = test_dataset[idx]
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"Category: {label:02d}") # 格式化為兩位數
        plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# 主程式執行流程
if __name__ == "__main__":
    # 1. 設定
    DEVICE = select_device()
    BATCH_SIZE = 300
    EPOCHS = 30 # 增加 Epoch 數量，因為任務更複雜
    LEARNING_RATE = 0.001
    MODEL_SAVE_PATH = "pytorch_cnn_2digits.pth" # 更改模型儲存路徑

    # 2. 載入並準備資料
    # 檢查雙位數 MNIST 資料集是否存在，如果不存在則生成
    double_mnist_data_path = './data/double_mnist'
    if not os.path.exists(os.path.join(double_mnist_data_path, 'train')):
        print(f"雙位數 MNIST 資料集 '{double_mnist_data_path}' 不存在，正在生成中...")
        # 執行 prepare_double_mnist.py 來生成資料集
        # 注意：這裡假設 prepare_double_mnist.py 位於當前工作目錄
        os.system("python prepare_double_mnist.py")
        print("雙位數 MNIST 資料集生成完成。")
    else:
        print(f"雙位數 MNIST 資料集 '{double_mnist_data_path}' 已存在。")

    train_loader, val_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE, data_root=double_mnist_data_path)
    
    # 為了視覺化，我們也載入一份未經轉換的測試資料
    raw_test_dataset = DoubleMNISTDataset(root_dir=os.path.join(double_mnist_data_path, 'test'), transform=transforms.ToTensor())

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
        patience=5,       # 5 個 epoch 沒改善就調整 (稍微增加耐心)
        min_lr=1e-6       # 設定最小學習率
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
    # 類別範圍從 0 到 99
    classes_range = [f"{i:02d}" for i in range(100)] # 格式化為兩位數字串
    plot_confusion_matrix(cm, classes=classes_range, title='Confusion Matrix (Two Digits)')
    plot_confusion_matrix(cm, classes=classes_range, normalize=True, title='Normalized Confusion Matrix (Two Digits)')

    # 7. 視覺化預測
    visualize_predictions(model, DEVICE, raw_test_dataset, y_test, y_pred)
    
    # 8. 儲存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"模型狀態已儲存至 {MODEL_SAVE_PATH}")