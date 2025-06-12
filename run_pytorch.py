# 檔名: run_pytorch.py
"""
手寫數字即時辨識程式 (PyTorch 版本)
使用攝影機捕捉手寫數字，並透過 PyTorch 模型進行即時辨識
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os

# --- 模型定義 ---
# 我們需要再次定義與訓練時完全相同的模型架構，以便正確載入權重。
# 這部分直接從 train_pytorch.py 複製過來。
class CNN(nn.Module):
    """
    卷積神經網路模型架構
    繼承自 torch.nn.Module
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        """定義模型的前向傳播路徑"""
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_pytorch_model(model_path, device):
    """
    載入預訓練的 PyTorch 模型

    Args:
        model_path (str): 模型權重檔案的路徑
        device (torch.device): 要將模型載入到的設備 (CPU 或 GPU)

    Returns:
        torch.nn.Module: 載入權重並設定為評估模式的模型
    """
    print('載入 PyTorch 模型中...')
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # *** 非常重要：將模型設定為評估模式 ***
    return model

def process_frame_pytorch(frame, model, device):
    """
    處理視訊幀並使用 PyTorch 模型進行數字辨識

    Args:
        frame: 從攝影機捕捉的原始影像幀
        model: 預訓練的 PyTorch 模型
        device: 計算設備

    Returns:
        frame: 處理後的影像幀，包含辨識結果
    """
    # 調整影像尺寸以加快處理效率
    frame = cv2.resize(frame, (540, 300))
    
    # 定義擷取數字的區域位置和大小 (ROI: Region of Interest)
    x, y, w, h = 400, 180, 120, 120
    
    # 複製 ROI 區域的影像以進行辨識
    img_roi = frame.copy()[y:y+h, x:x+w]
    
    # --- 預處理影像 ---
    # 1. 顏色轉成灰階
    img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    
    # 2. 針對白色文字，做二值化黑白轉換，轉成黑底白字
    _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 3. 將處理後的黑白影像顯示在畫面右上角
    output_display = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)
    frame[0:120, 420:540] = output_display
    
    # 4. 準備影像進行辨識 (這是關鍵步驟)
    #    預處理必須和訓練時完全一致！
    img_resized = cv2.resize(img_binary, (28, 28))

    # 定義與訓練時相同的轉換
    preprocess = transforms.Compose([
        transforms.ToTensor(), # 將 NumPy 陣列轉為 Tensor，並標準化到 [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)) # 使用 MNIST 的平均值和標準差再次標準化
    ])
    
    img_tensor = preprocess(img_resized)
    
    # 5. 增加批次維度 (batch dimension) 並移至指定設備
    #    模型預期的輸入是 (N, C, H, W)，目前是 (C, H, W)
    input_tensor = img_tensor.unsqueeze(0).to(device)

    # --- 進行辨識 ---
    with torch.no_grad(): # 在推論時不計算梯度，以節省資源
        prediction = model(input_tensor)
        predicted_prob = F.softmax(prediction, dim=1)
        predicted_digit = torch.argmax(predicted_prob).item()

    # --- 在影像上顯示辨識結果 ---
    cv2.putText(
        frame,
        str(predicted_digit),
        (x, y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        2, (0, 0, 255), 2, cv2.LINE_AA
    )
    
    # 標記辨識區域
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
    
    return frame

def main():
    """主程式：啟動攝影機並進行即時手寫數字辨識"""
    MODEL_PATH = "pytorch_cnn.pth"
    
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：找不到模型檔案 '{MODEL_PATH}'")
        print("請先執行 train_pytorch.py 來訓練並儲存模型。")
        return

    # 選擇設備
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"將使用 {DEVICE} 進行辨識。")
    
    # 載入模型
    model = load_pytorch_model(MODEL_PATH, DEVICE)
    print('模型載入完成，開始辨識...')
    
    # 啟用攝影鏡頭
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("無法開啟攝影機")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法接收影像幀")
                break
            
            # 處理影像並進行辨識
            processed_frame = process_frame_pytorch(frame, model, DEVICE)
            
            # 顯示結果
            cv2.imshow('PyTorch Handwritten Digits Recognition', processed_frame)
            
            # 按下 'q' 鍵停止
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 釋放資源
        cap.release()
        cv2.destroyAllWindows()
        print("程式已結束。")

# 執行主程式
if __name__ == "__main__":
    main()