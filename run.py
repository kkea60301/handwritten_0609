"""
手寫數字即時辨識程式
使用攝影機捕捉手寫數字並進行即時辨識
"""

import cv2                              # 電腦視覺處理
import numpy as np                      # 數值計算和矩陣操作
import tensorflow as tf                 # 深度學習框架
from keras.models import Sequential     # 序列式模型架構
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # 神經網路層


def load_model():
    """載入預訓練的手寫數字辨識模型
    
    Returns:
        model: 載入權重後的模型
    """
    print('載入模型中...')
    
    # 建立與訓練時相同的模型架構
    model = Sequential()
    
    # 第一個卷積層和池化層
    model.add(Conv2D(
        32, (5, 5), 
        activation="relu", 
        padding="same", 
        data_format="channels_last", 
        input_shape=(28, 28, 1)
    ))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
    
    # 第二個卷積層和池化層
    model.add(Conv2D(
        32, (5, 5), 
        activation="relu", 
        padding="same", 
        data_format="channels_last"
    ))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
    
    # 全連接層
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    
    # 載入預訓練權重
    model.load_weights('cnn2.weights.h5')
    
    return model


def process_frame(frame, model):
    """處理視訊幀並進行數字辨識
    
    Args:
        frame: 從攝影機捕捉的原始影像幀
        model: 預訓練的手寫數字辨識模型
        
    Returns:
        frame: 處理後的影像幀，包含辨識結果
    """
    # 調整影像尺寸以加快處理效率
    frame = cv2.resize(frame, (540, 300))
    
    # 定義擷取數字的區域位置和大小
    x, y, w, h = 400, 180, 120, 120
    
    # 複製一個影像作為辨識使用
    img_num = frame.copy()
    img_num = img_num[y:y+h, x:x+w]
    
    # 顏色轉成灰階
    img_num = cv2.cvtColor(img_num, cv2.COLOR_BGR2GRAY)
    
    # 針對白色文字，做二值化黑白轉換，轉成黑底白字
    _, img_num = cv2.threshold(img_num, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 將轉換後的影像顯示在畫面右上角
    output = cv2.cvtColor(img_num, cv2.COLOR_GRAY2BGR)
    frame[0:120, 420:540] = output
    
    # 準備影像進行辨識
    img_for_prediction = cv2.resize(img_num, (28, 28))
    img_for_prediction = img_for_prediction.astype(np.float32)
    img_for_prediction = np.expand_dims(img_for_prediction, axis=0)  # 增加批次維度
    img_for_prediction = np.expand_dims(img_for_prediction, axis=3)  # 增加通道維度
    img_for_prediction = img_for_prediction / 255.0  # 標準化
    
    # 進行辨識
    prediction = model.predict(img_for_prediction)
    predicted_digit = str(np.argmax(prediction))
    
    # 在影像上顯示辨識結果
    cv2.putText(
        frame,                          # 目標影像
        predicted_digit,                # 顯示文字
        (x, y-20),                      # 文字位置
        cv2.FONT_HERSHEY_SIMPLEX,       # 字體
        2,                              # 字體大小
        (0, 0, 255),                    # 顏色 (BGR)
        2,                              # 線條粗細
        cv2.LINE_AA                     # 線條類型
    )
    
    # 標記辨識區域
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
    
    return frame


def main():
    """主程式：啟動攝影機並進行即時手寫數字辨識"""
    # 載入模型
    model = load_model()
    print('模型載入完成，開始辨識...')
    
    # 啟用攝影鏡頭
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("無法開啟攝影機")
        return
    
    try:
        while True:
            # 讀取一幀影像
            ret, frame = cap.read()
            if not ret:
                print("無法接收影像幀")
                break
            
            # 處理影像並進行辨識
            processed_frame = process_frame(frame, model)
            
            # 顯示結果
            cv2.imshow('Handwritten_digits', processed_frame)
            
            # 按下 q 鍵停止
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    finally:
        # 釋放資源
        cap.release()
        cv2.destroyAllWindows()


# 執行主程式
if __name__ == "__main__":
    main()