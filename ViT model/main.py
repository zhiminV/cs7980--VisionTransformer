import torch
import torch.nn as nn
import tensorflow as tf
from train import train_model
from evaluate import evaluate_model
from visionTransformer import VisionTransformer
from preprocess_data import get_dataset
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

# Configuration for ViT model
config = {
    'image_height': 64,
    'image_width': 64,
    'im_channels': 12,
    'emb_dim': 768,
    'patch_emb_drop': 0.1,
    'patch_height': 4,
    'patch_width': 4,
    'num_heads': 8,
    'mlp_dim': 2048,
    'num_layers': 12,
    'dropout_rate': 0.1
}

# Constants
DATA_SIZE = 64
PATCH_SIZE = 64
BATCH_SIZE = 4 # 根据GPU 内存来调size， 4090 24G内存是可以跑 BATCH_SIZE = 8，Bronte 是32
NUM_CLASSES = 1
EPOCHS = 15 # 根据需求改变
LEARNING_RATE = 0.001

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit= 50000)])  # 配置 TensorFlow 在使用 GPU 时的内存限制
    except RuntimeError as e:
        print(e)

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vit_model = VisionTransformer(config, num_classes=NUM_CLASSES)

# Load datasets
train_dataset = get_dataset(
    file_pattern='/home/liang.zhimi/ondemand/northamerica_2012-2023/train/*_ongoing_*.tfrecord',
    data_size=DATA_SIZE,
    sample_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    num_in_channels=12,
    compression_type=None,
    clip_and_normalize=True,
    clip_and_rescale=False,
    random_crop=True,
    center_crop=False
)

validation_dataset = get_dataset(
    file_pattern='/home/liang.zhimi/ondemand/northamerica_2012-2023/val/*_ongoing_*.tfrecord',
    data_size=DATA_SIZE,
    sample_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    num_in_channels=12,
    compression_type=None,
    clip_and_normalize=True,
    clip_and_rescale=False,
    random_crop=True,
    center_crop=False
)

test_dataset = get_dataset(
    file_pattern='/home/liang.zhimi/ondemand/northamerica_2012-2023/test/*_ongoing_*.tfrecord',
    data_size=DATA_SIZE,
    sample_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    num_in_channels=12,
    compression_type=None,
    clip_and_normalize=True,
    clip_and_rescale=False,
    random_crop=True,
    center_crop=False
)

# Train the ViT model
train_losses, val_losses = train_model(vit_model, train_dataset, validation_dataset, epochs=EPOCHS, device=device)

# Plot loss functions...
def plot_train_and_val_losses(train_losses, val_losses):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(train_losses)
    axs[0].set_title("train loss")
    axs[1].plot(val_losses)
    axs[1].set_title("validation loss")
    plt.show()

plot_train_and_val_losses(train_losses, val_losses)

# Load the best model for testing
vit_model.load_state_dict(torch.load("vit_model.pth"))
vit_model.to(device)

print("Evaluation...")
print("Test set metrics:")
IoU, recall, precision, val_loss = evaluate_model(lambda x: torch.sigmoid(vit_model(x)).detach().cpu().numpy(), test_dataset, device=device)
print(f"Mean IoU: {IoU}\nMean precision: {precision}\nMean recall: {recall}\nTest loss: {val_loss}")

# Inference...
def show_inference(n_rows: int, features: tf.Tensor, label: tf.Tensor, prediction_function):
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)
    
    fig = plt.figure(figsize=(15, n_rows * 4))
    
    # 确保传递给 prediction_function 的是 PyTorch Tensor
    features = features.permute(0, 2, 3, 1)  # 将特征张量调整为 (batch, height, width, channels)
    prediction = prediction_function(features)

    for i in range(n_rows):
        plt.subplot(n_rows, 3, i * 3 + 1)
        plt.title("Previous day fire")
        feature_img = features[i, :, :, -1].cpu().numpy()  # 转换为 NumPy 数组
        plt.imshow(feature_img, cmap=CMAP, norm=NORM)
        plt.axis('off')

        plt.subplot(n_rows, 3, i * 3 + 2)
        plt.title("True next day fire")
        label_img = label[i, 0, :, :].cpu().numpy()   # 转换为 NumPy 数组
        plt.imshow(label_img, cmap=CMAP, norm=NORM)
        plt.axis('off')
    
        plt.subplot(n_rows, 3, i * 3 + 3)
        plt.title("Predicted next day fire")
        pred_img = prediction[i, 0, :, :]  # 确保是二维数组
        plt.imshow(pred_img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

features, labels = next(iter(test_dataset))
# 将 TensorFlow 张量转换为 PyTorch 张量
features_torch = torch.from_numpy(features.numpy()).permute(0, 3, 1, 2).float().to(device)
labels_torch = torch.from_numpy(labels.numpy()).permute(0, 3, 1, 2).float().to(device)

# 调用 show_inference 函数
show_inference(5, features_torch, labels_torch, lambda x: torch.sigmoid(vit_model(x.permute(0, 3, 1, 2))).detach().cpu().numpy())