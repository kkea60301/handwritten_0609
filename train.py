"""
手寫數字識別模型訓練程式
使用 TensorFlow 和 Keras 建立卷積神經網路 (CNN) 模型
"""

# 載入必要的函式庫
import numpy as np                      # 數值計算和矩陣操作
import matplotlib.pyplot as plt         # 資料視覺化和繪圖
import tensorflow as tf                 # 深度學習框架
from keras.datasets import mnist        # MNIST 手寫數字資料集
from keras.utils import to_categorical  # 將標籤轉換為 one-hot 編碼
from keras.models import Sequential     # 序列式模型架構
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D  # 神經網路層
from keras.optimizers import Adam       # Adam 優化器
from keras.preprocessing.image import ImageDataGenerator  # 資料增強
from keras.callbacks import ReduceLROnPlateau  # 學習率調整回調
from sklearn.model_selection import train_test_split  # 資料分割
from sklearn.metrics import confusion_matrix  # 混淆矩陣
import itertools                        # 用於混淆矩陣繪製

# 設定隨機種子，確保實驗可重現
np.random.seed(42)


def configure_gpu():
    """配置 TensorFlow 以使用 GPU
    
    列出所有可用的 GPU 設備，讓使用者選擇要使用的 GPU，
    並設定記憶體增長以避免佔用全部 GPU 記憶體
    """
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
        print("偵測到以下 GPU 設備：")
        for i, gpu in enumerate(physical_gpus):
            print(f"  {i}: {gpu.name}")
        
        while True:
            try:
                choice = int(input("請輸入您想使用的 GPU Index (例如 0, 1, ...): "))
                if 0 <= choice < len(physical_gpus):
                    selected_gpu = physical_gpus[choice]
                    tf.config.set_visible_devices(selected_gpu, 'GPU')
                    tf.config.experimental.set_memory_growth(selected_gpu, True)
                    logical_gpus = tf.config.list_logical_devices('GPU')
                    print(f"已選擇 GPU {choice} ({selected_gpu.name})。")
                    print(f"偵測到 {len(physical_gpus)} 個實體 GPU, {len(logical_gpus)} 個邏輯 GPU。")
                    print("訓練將在 GPU 上運行。")
                    break
                else:
                    print("無效的選擇，請輸入正確的 GPU Index。")
            except ValueError:
                print("無效的輸入，請輸入一個數字。")
            except RuntimeError as e:
                print(f"GPU 配置失敗: {e}。訓練將在 CPU 上運行。")
                break
    else:
        print("未偵測到 GPU 設備。訓練將在 CPU 上運行。")


def load_and_prepare_data():
    """載入 MNIST 資料集並進行預處理
    
    包括資料分割、視覺化樣本、增加通道維度和標準化
    
    Returns:
        tuple: 包含處理後的訓練、驗證和測試資料及其標籤
    """
    # 載入 MNIST 資料集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 將訓練集分割為訓練集和驗證集
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=25
    )
    
    # 視覺化部分訓練樣本
    plt.figure(figsize=(4, 4))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(x_train[i], cmap="gray")
        plt.title(f"Class {y_train[i]}")
    plt.tight_layout()
    plt.show()
    
    # 增加通道維度 (灰階圖像為單通道)
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    x_val = np.expand_dims(x_val, axis=3)
    
    # 顯示資料形狀
    print(f"訓練資料形狀: {x_train.shape}")
    print(f"測試資料形狀: {x_test.shape}")
    
    # 標準化像素值到 [0,1] 範圍
    x_train_normalized = x_train / 255.0
    x_test_normalized = x_test / 255.0
    x_val_normalized = x_val / 255.0
    
    return (
        x_train_normalized, y_train, 
        x_val_normalized, y_val, 
        x_test_normalized, y_test,
        x_test, y_test  # 保留未標準化的測試資料用於評估
    )


def show_train_history(train_history, train, validation):
    """繪製訓練歷史圖表
    
    Args:
        train_history: 模型訓練的歷史記錄
        train: 訓練指標名稱 (如 'accuracy', 'loss')
        validation: 驗證指標名稱 (如 'val_accuracy', 'val_loss')
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_history.history[train], label=f'Train {train}')
    plt.plot(train_history.history[validation], label=f'Validation {validation}')
    plt.title("Training History")
    plt.ylabel(train.capitalize())
    plt.xlabel('Train cycles (Epoch)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def create_model():
    """建立卷積神經網路模型
    
    Returns:
        model: 建立好的 Keras 序列模型
    """
    # 嘗試在 GPU 上建立模型
    with tf.device('/GPU:0'):
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
    model.add(Dropout(0.5))  # 防止過擬合
    model.add(Dense(10, activation="softmax"))  # 輸出層，10個類別
    
    # 顯示模型摘要
    model.summary()
    
    return model


def prepare_training(model, x_train_normalized, y_train, x_test_normalized, y_test):
    """準備模型訓練
    
    編譯模型、準備資料增強和回調函數
    
    Args:
        model: 要訓練的模型
        x_train_normalized: 標準化的訓練資料
        y_train: 訓練標籤
        x_test_normalized: 標準化的測試資料
        y_test: 測試標籤
        
    Returns:
        tuple: 包含訓練生成器、測試生成器和學習率調整回調
    """
    # 編譯模型
    model.compile(
        loss='categorical_crossentropy',  # 多分類問題的標準損失函數
        optimizer=Adam(),                 # Adam 優化器
        metrics=['accuracy']              # 評估指標
    )
    
    # 將標籤轉換為 one-hot 編碼
    y_train_onehot = to_categorical(y_train)
    
    # 設定資料增強 - 用於訓練集
    train_datagen = ImageDataGenerator(
        rotation_range=8,           # 隨機旋轉角度範圍
        width_shift_range=0.08,     # 水平平移範圍
        height_shift_range=0.08,    # 垂直平移範圍
        shear_range=0.3,            # 剪切變換範圍
        zoom_range=0.08,            # 縮放範圍
        data_format="channels_last"
    )
    
    # 設定測試資料生成器 - 不進行增強
    test_datagen = ImageDataGenerator(data_format="channels_last")
    
    # 準備資料生成器
    train_datagen.fit(x_train_normalized)
    train_generator = train_datagen.flow(
        x_train_normalized, 
        y_train_onehot, 
        batch_size=300
    )
    
    test_datagen.fit(x_test_normalized)
    test_generator = test_datagen.flow(
        x_test_normalized, 
        y_test, 
        batch_size=300
    )
    
    # 設定學習率調整回調
    learning_rate_callback = ReduceLROnPlateau(
        monitor='val_accuracy',  # 監控驗證準確率
        patience=3,              # 3個 epoch 無改善則調整學習率
        verbose=1,               # 顯示調整訊息
        factor=0.5,              # 學習率減半
        min_lr=0.00001           # 最小學習率
    )
    
    return train_generator, test_generator, learning_rate_callback


def train_model(model, train_generator, x_val_normalized, y_val, learning_rate_callback):
    """訓練模型
    
    Args:
        model: 要訓練的模型
        train_generator: 訓練資料生成器
        x_val_normalized: 標準化的驗證資料
        y_val: 驗證標籤
        learning_rate_callback: 學習率調整回調
        
    Returns:
        train_history: 訓練歷史記錄
    """
    # 開始訓練
    train_history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=30,
        verbose=1,
        validation_data=(x_val_normalized, to_categorical(y_val)),
        callbacks=[learning_rate_callback]
    )
    
    return train_history


def evaluate_model(model, x_test, y_test, train_history):
    """評估模型性能
    
    計算測試集上的準確率並獲取預測結果
    
    Args:
        model: 訓練好的模型
        x_test: 測試資料
        y_test: 測試標籤
        train_history: 訓練歷史記錄
        
    Returns:
        tuple: 包含準確率和預測結果
    """
    # 顯示訓練歷史
    show_train_history(train_history, 'accuracy', 'val_accuracy')
    show_train_history(train_history, 'loss', 'val_loss')
    
    # 在測試集上評估模型
    y_test_onehot = to_categorical(y_test)
    score = model.evaluate(x_test, y_test_onehot)
    print("\n測試集準確率: {:.2f}%".format(score[1] * 100))
    
    # 獲取預測結果
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=-1)
    
    return score[1], predicted_classes


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    """繪製混淆矩陣
    
    Args:
        cm: 混淆矩陣
        classes: 類別標籤
        normalize: 是否標準化
        title: 圖表標題
        cmap: 顏色映射
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    
    # 設定座標軸
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # 標準化混淆矩陣
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 在每個格子中顯示數值
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Real label', fontsize=12)
    plt.xlabel('Predict label', fontsize=12)
    plt.grid(False)
    plt.show()


def analyze_results(y_test, predicted_classes):
    """分析模型預測結果
    
    生成並顯示混淆矩陣
    
    Args:
        y_test: 測試標籤
        predicted_classes: 預測類別
    """
    # 計算混淆矩陣
    cm = confusion_matrix(y_test, predicted_classes)
    
    # 繪製混淆矩陣
    plot_confusion_matrix(cm, range(10))  # 10 個類別 (0-9)
    
    # 也可以繪製標準化的混淆矩陣
    plot_confusion_matrix(cm, range(10), normalize=True, title="Normalized Confusion Matrix")


def visualize_predictions(x_test, y_test, predicted_classes):
    """視覺化預測結果
    
    顯示一些錯誤和正確的預測範例
    
    Args:
        x_test: 測試資料
        y_test: 測試標籤
        predicted_classes: 預測類別
    """
    # 準備資料
    test_set = np.squeeze(x_test, axis=3)  # 移除通道維度以便顯示
    
    # 找出錯誤預測的樣本
    incorrect_indices = np.where(y_test != predicted_classes)[0]
    
    # 顯示錯誤預測的樣本
    plt.figure(figsize=(10, 10))
    plt.suptitle("Incorrect Indices", fontsize=16)
    for i in range(9):
        if i < len(incorrect_indices):
            plt.subplot(3, 3, i+1)
            idx = incorrect_indices[i]
            img = test_set[idx]
            plt.imshow(img, cmap="gray")
            plt.title(f"Real: {y_test[idx]} / Predict: {predicted_classes[idx]}")
            plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    # 找出正確預測的樣本
    correct_indices = np.where(y_test == predicted_classes)[0]
    
    # 顯示正確預測的樣本
    plt.figure(figsize=(10, 10))
    plt.suptitle("Correct indices", fontsize=16)
    for i in range(9):
        if i < len(correct_indices):
            plt.subplot(3, 3, i+1)
            idx = correct_indices[i]
            img = test_set[idx]
            plt.imshow(img, cmap="gray")
            plt.title(f"Category: {y_test[idx]}")
            plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# 主程式執行流程
if __name__ == "__main__":
    # 調用 GPU 配置函數
    configure_gpu()
    
    # 載入並準備資料
    x_train_normalized, y_train, x_val_normalized, y_val, x_test_normalized, y_test, x_test, y_test = load_and_prepare_data()
    
    # 建立模型
    model = create_model()
    
    # 準備訓練
    train_generator, test_generator, learning_rate_callback = prepare_training(
        model, x_train_normalized, y_train, x_test_normalized, y_test
    )
    
    # 訓練模型
    train_history = train_model(
        model, train_generator, x_val_normalized, y_val, learning_rate_callback
    )
    
    # 評估模型
    accuracy, predicted_classes = evaluate_model(model, x_test, y_test, train_history)
    
    # 分析結果
    analyze_results(y_test, predicted_classes)
    
    # 視覺化預測
    visualize_predictions(x_test, y_test, predicted_classes)
    
    # 儲存模型權重
    model.save_weights("cnn2.weights.h5")
    print("模型權重已儲存至 cnn2.weights.h5")
