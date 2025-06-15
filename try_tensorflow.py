"""
GPU 環境檢測程式
用於檢查 TensorFlow 是否能夠使用 GPU 加速
"""

# 載入必要的函式庫
import numpy as np                      # 數值計算和矩陣操作
import matplotlib.pyplot as plt         # 資料視覺化和繪圖
import tensorflow as tf                 # 深度學習框架
from keras.datasets import mnist        # MNIST 手寫數字資料集
from sklearn.model_selection import train_test_split  # 資料分割

# 設定隨機種子，確保實驗可重現
np.random.seed(42)


def check_gpu_availability():
    """檢查 GPU 可用性
    
    列出所有可用的 GPU 設備並顯示是否使用 GPU 進行計算
    """
    # 列出所有可用的物理設備
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"可用的 GPU 數量: {len(physical_devices)}")

    # 確認是否使用 GPU 進行計算
    if physical_devices:
        print("使用 GPU 進行計算")
        for i, gpu in enumerate(physical_devices):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print("使用 CPU 進行計算")


# 執行 GPU 檢查
if __name__ == "__main__":
    check_gpu_availability()
