# 檔名: run_pytorch_shot.py
"""
手寫數字影像辨識程式 (PyTorch 版本)
使用相機捕捉手寫數字照片，並透過 PyTorch 模型進行辨識
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt

# 全域變數，用於儲存選取的 ROI 座標 (x, y, w, h)
g_roi = None
# 全域變數，用於判斷是否正在拖曳滑鼠
g_drawing = False
# 全域變數，用於儲存滑鼠拖曳的起始點
g_start_point = (-1, -1)
# 全域變數，用於儲存拍照的影像，供滑鼠回呼函式使用
g_temp_frame = None

# 預設 ROI 座標 (針對 540x300 影像)
DEFAULT_ROI_X = 400
DEFAULT_ROI_Y = 180
DEFAULT_ROI_W = 120
DEFAULT_ROI_H = 120

# 視窗名稱常數
MAIN_WINDOW_NAME = 'Digits Recognition - Press "s" to Shot, "q" to Quit'
CAPTURED_WINDOW_NAME = 'Handwritten Digits Recognition - Captured Image'
RESULT_WINDOW_NAME = 'PyTorch Handwritten Digits Recognition - Result'

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

def mouse_callback(event, x, y, flags, param):
    """
    滑鼠事件回呼函式，用於選擇 ROI
    """
    global g_roi, g_drawing, g_start_point, g_temp_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        g_drawing = True
        g_start_point = (x, y)
        g_roi = None # 重置 ROI

    elif event == cv2.EVENT_MOUSEMOVE:
        if g_drawing:
            # 繪製即時選取框
            temp_frame_display = g_temp_frame.copy() # 使用 g_temp_frame 的副本進行繪製
            cv2.rectangle(temp_frame_display, g_start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow(CAPTURED_WINDOW_NAME, temp_frame_display)

    elif event == cv2.EVENT_LBUTTONUP:
        g_drawing = False
        end_point = (x, y)
        # 確保座標是正確的 (左上角, 右下角)
        x1, y1 = min(g_start_point[0], end_point[0]), min(g_start_point[1], end_point[1])
        x2, y2 = max(g_start_point[0], end_point[0]), max(g_start_point[1], end_point[1])
        
        # 確保選取區域有效
        if x2 - x1 > 0 and y2 - y1 > 0:
            g_roi = (x1, y1, x2 - x1, y2 - y1)
            print(f"選取區域: {g_roi}")
        else:
            g_roi = None # 無效選取

def process_frame_pytorch(frame, model, device, roi=None):
    """
    處理視訊幀並使用 PyTorch 模型進行數字辨識

    Args:
        frame: 從相機捕捉的原始影像幀
        model: 預訓練的 PyTorch 模型
        device: 計算設備
        roi (tuple, optional): 選取的 ROI 座標 (x, y, w, h)。如果為 None，則使用預設 ROI。

    Returns:
        frame: 處理後的影像幀，包含辨識結果
    """
    
    if roi:
        x, y, w, h = roi
    else:
        # 定義擷取數字的區域位置和大小 (ROI: Region of Interest)
        x, y, w, h = DEFAULT_ROI_X, DEFAULT_ROI_Y, DEFAULT_ROI_W, DEFAULT_ROI_H # 這些值是針對 540x300 影像的
    
    # 複製 ROI 區域的影像以進行辨識
    # 確保 ROI 區域在 frame 的範圍內
    img_roi = frame.copy()[max(0, y):min(frame.shape[0], y+h), max(0, x):min(frame.shape[1], x+w)]
    
    # 如果 img_roi 為空，表示選取區域無效，直接返回原始 frame
    if img_roi.shape[0] == 0 or img_roi.shape[1] == 0:
        print("選取區域無效，無法進行辨識。")
        return frame

    # --- 預處理影像 ---
    # 1. 顏色轉成灰階
    img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    
    # 2. 針對白色文字，做二值化黑白轉換，轉成黑底白字
    _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 3. 將處理後的黑白影像顯示在畫面右上角
    output_display = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)
    # 確保右上角顯示區域不會超出畫面
    display_x_offset = frame.shape[1] - output_display.shape[1]
    display_y_offset = 0
    # 確保複製區域大小匹配
    target_h, target_w = output_display.shape[0], output_display.shape[1]
    frame[display_y_offset : display_y_offset + target_h,
          display_x_offset : display_x_offset + target_w] = output_display
    
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
        
        # 顯示機率分佈長條圖
        draw_probability_bar_chart(predicted_prob)

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

def draw_probability_bar_chart(probabilities):
    """
    繪製機率分佈的長條圖

    Args:
        probabilities (torch.Tensor): 模型的輸出機率分佈 (1x10 的 Tensor)
    """
    # 將 Tensor 轉換為 NumPy 陣列
    probs_np = probabilities.cpu().numpy().flatten()

    # 建立數字標籤 (0-9)
    labels = [str(i) for i in range(10)]

    # 建立長條圖
    plt.figure(figsize=(8, 5))
    plt.bar(labels, probs_np, color='skyblue')
    plt.xlabel('Digits')
    plt.ylabel('Probability')
    plt.title('Digit Recognition Probability Distribution')
    plt.ylim(0, 1) # 機率範圍在 0 到 1 之間
    
    # 在每個長條上方顯示機率值
    for i, prob in enumerate(probs_np):
        plt.text(i, prob + 0.02, f'{prob:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def is_window_closed(window_name):
    """檢查指定的 OpenCV 視窗是否已關閉"""
    return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1

def list_available_cameras():
    """
    列出所有可用的相機裝置及其編號。
    返回一個包含可用相機編號的列表。
    """
    available_cameras = []
    # 嘗試開啟從 0 到 9 的相機編號
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release() # 釋放相機
    return available_cameras

def main():
    """主程式：啟動相機，拍照後進行手寫數字辨識"""
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
    print('模型載入完成，等待拍照...')
    
    # 列出可用的相機並讓使用者選擇
    available_cameras = list_available_cameras()
    if not available_cameras:
        print("錯誤：找不到任何可用的相機。請確認相機已正確連接。")
        return

    camera_index = 0
    if len(available_cameras) > 1:
        print("偵測到多個相機。以下為相機編號：")
        for i, cam_idx in enumerate(available_cameras):
            print(f"  {i+1}. 相機 {cam_idx}")
        
        while True:
            try:
                choice = int(input("請輸入要使用的相機編號 (例如：1, 2, ...): "))
                if 1 <= choice <= len(available_cameras):
                    camera_index = available_cameras[choice - 1]
                    break
                else:
                    print("無效的選擇，請重新輸入。")
            except ValueError:
                print("無效的輸入，請輸入數字。")
    else:
        print(f"偵測到一個相機：相機編號 {available_cameras[0]}")
        camera_index = available_cameras[0]

    # 啟用攝影鏡頭
    cap = cv2.VideoCapture(camera_index)

    # 設定相機解析度為 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # 設定幀率為 30fps
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 創建主視窗並設定為可調整大小
    cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    if not cap.isOpened():
        print(f"無法開啟相機 {camera_index}")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法接收影像幀")
            cap.release()
            cv2.destroyAllWindows()
            break
        
        # 獲取當前主視窗的尺寸
        # getWindowImageRect 返回 (x, y, width, height)
        x, y, window_width, window_height = cv2.getWindowImageRect(MAIN_WINDOW_NAME)

        # 如果視窗尺寸有效 (寬高大於0)，則根據視窗大小調整影像尺寸
        # 注意：這裡不再強制縮放為 540x300，而是使用實際的相機解析度
        if window_width > 0 and window_height > 0:
            # 確保 frame_display 的尺寸與視窗尺寸匹配，避免顯示問題
            frame_display = cv2.resize(frame, (window_width, window_height))
        else:
            # 否則，使用相機的實際解析度
            frame_display = frame.copy()

        # 顯示即時影像
        cv2.imshow(MAIN_WINDOW_NAME, frame_display)
        
        # 檢查主視窗是否被關閉
        if is_window_closed(MAIN_WINDOW_NAME):
            print("主視窗已關閉。")
            cap.release()
            cv2.destroyAllWindows()
            break

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'): # 按下 's' 鍵拍照
            print("已拍照，請在影像上選取辨識區域...")
            global g_roi, g_temp_frame # 宣告為 global
            g_roi = None # 每次拍照時重置 ROI
            g_temp_frame = frame_display.copy() # 儲存拍照的影像 (已縮放)，用於繪製選取框
            
            # 進入選取區域模式
            while True:
                cv2.imshow(CAPTURED_WINDOW_NAME, g_temp_frame)
                cv2.setMouseCallback(CAPTURED_WINDOW_NAME, mouse_callback)
                
                # 檢查選取視窗是否被關閉
                if is_window_closed(CAPTURED_WINDOW_NAME):
                    print("影像選取視窗已關閉。")
                    break # 跳出選取區域迴圈
                
                key_select = cv2.waitKey(1) & 0xFF
                
                if g_roi is not None: # 如果已選取區域
                    print("已選取區域，進行辨識...")
                    processed_frame = process_frame_pytorch(g_temp_frame.copy(), model, DEVICE, roi=g_roi) # 傳入 g_temp_frame 的副本
                    cv2.imshow(RESULT_WINDOW_NAME, processed_frame)
                    
                    # 等待使用者操作：重新選取或返回主畫面
                    while True:
                        # 檢查結果視窗是否被關閉
                        if is_window_closed(RESULT_WINDOW_NAME):
                            print("辨識結果視窗已關閉。")
                            break # 跳出結果顯示迴圈
                        
                        result_key = cv2.waitKey(1) & 0xFF
                        if result_key == ord('r'): # 按下 'r' 鍵重新選取
                            print("重新選取區域...")
                            g_roi = None # 重置 ROI，以便重新選取
                            cv2.destroyWindow(RESULT_WINDOW_NAME)
                            # 重新顯示 captured image 視窗，以便重新選取
                            cv2.imshow(CAPTURED_WINDOW_NAME, g_temp_frame)
                            cv2.setMouseCallback(CAPTURED_WINDOW_NAME, mouse_callback)
                            break # 跳出結果顯示迴圈，回到選取區域迴圈
                        elif result_key == ord('q'): # 按下 'q' 鍵退出
                            print("程式結束。")
                            cap.release()
                            cv2.destroyAllWindows()
                            return # 結束主程式
                    
                    if result_key == ord('q'): # 如果在結果視窗按了 'q'，則直接退出
                        break # 跳出選取區域迴圈
                    
                elif key_select == ord('c'): # 按下 'c' 鍵取消選取
                    print("取消選取。")
                    break # 跳出選取區域迴圈
            
            cv2.destroyWindow(CAPTURED_WINDOW_NAME)
            cv2.destroyWindow(RESULT_WINDOW_NAME) # 確保關閉所有相關視窗
            print("辨識完成或取消，等待下一次拍照或退出...")
            
        elif key == ord('q'): # 按下 'q' 鍵停止
            print("程式結束。")
            cap.release()
            cv2.destroyAllWindows()
            break

# 執行主程式
if __name__ == "__main__":
    main()
