import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import numpy as np

def create_double_mnist_dataset(root_dir='./data', output_dir='./data/double_mnist', num_samples_per_digit=1000):
    """
    從原始 MNIST 數據集創建雙位數 MNIST 數據集。
    每個雙位數組合 (00-99) 將生成 num_samples_per_digit 個樣本。
    """
    print(f"正在準備雙位數 MNIST 數據集，輸出目錄：{output_dir}")

    # 確保輸出目錄存在
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    # 下載 MNIST 數據集
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root=root_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=root_dir, train=False, download=True, transform=transform)

    # 將數據集轉換為列表，方便隨機取樣
    train_images = [img for img, _ in mnist_train]
    train_labels = [label for _, label in mnist_train]
    test_images = [img for img, _ in mnist_test]
    test_labels = [label for _, label in mnist_test]

    # 創建訓練、驗證、測試數據
    # 這裡簡化處理，直接從訓練集和測試集生成雙位數圖像
    # 實際應用中可能需要更精細的劃分
    
    # 訓練集
    print("正在生成訓練集...")
    generate_double_images(train_images, train_labels, os.path.join(output_dir, 'train'), num_samples_per_digit)
    print("訓練集生成完成。")

    # 驗證集 (從訓練集中取一部分)
    print("正在生成驗證集...")
    generate_double_images(train_images, train_labels, os.path.join(output_dir, 'val'), num_samples_per_digit // 5) # 驗證集數量為訓練集的1/5
    print("驗證集生成完成。")

    # 測試集
    print("正在生成測試集...")
    generate_double_images(test_images, test_labels, os.path.join(output_dir, 'test'), num_samples_per_digit)
    print("測試集生成完成。")

    print("雙位數 MNIST 數據集準備完成！")

def generate_double_images(images, labels, output_path, num_samples):
    count = 0
    for i in range(10):
        for j in range(10):
            target_label = f"{i}{j}"
            label_dir = os.path.join(output_path, target_label)
            os.makedirs(label_dir, exist_ok=True)

            # 隨機選擇兩個圖像
            indices1 = [k for k, l in enumerate(labels) if l == i]
            indices2 = [k for k, l in enumerate(labels) if l == j]

            if not indices1 or not indices2:
                print(f"警告：數字 {i} 或 {j} 的圖像不足，跳過組合 {target_label}")
                continue

            for _ in range(num_samples):
                idx1 = np.random.choice(indices1)
                idx2 = np.random.choice(indices2)

                img1 = images[idx1].squeeze().numpy() * 255
                img2 = images[idx2].squeeze().numpy() * 255

                # 將兩個圖像水平拼接
                combined_img_np = np.concatenate((img1, img2), axis=1)
                combined_img = Image.fromarray(combined_img_np.astype(np.uint8))

                img_filename = os.path.join(label_dir, f"{count:05d}.png")
                combined_img.save(img_filename)
                count += 1
    print(f"在 {output_path} 中生成了 {count} 個雙位數圖像。")


if __name__ == '__main__':
    create_double_mnist_dataset()