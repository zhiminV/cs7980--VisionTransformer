import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import INPUT_FEATURES,OUTPUT_FEATURES, _get_features_dict

# 定义文件路径和数据集参数
file_pattern = '/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/train/*_ongoing_*.tfrecord'
data_size = 64  # 假设图像块的大小为64x64
sample_size = 64  # 模型输入的图像块大小
num_in_channels = len(INPUT_FEATURES)  # 输入特征的数量
batch_size = 1  # 每次读取一个示例

# 读取和解析TFRecord文件
def parse_example(example_proto):
    features_dict = _get_features_dict(data_size, INPUT_FEATURES + OUTPUT_FEATURES)
    example = tf.io.parse_single_example(example_proto, features_dict)
    input_features = [example[key] for key in INPUT_FEATURES]
    stacked_features = tf.stack(input_features, axis=-1)  # 将输入特征堆叠成多通道图像
    return stacked_features

# 获取数据集
dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(file_pattern))
dataset = dataset.map(parse_example)
dataset = dataset.batch(batch_size)

# 显示前几个图像
def display_images(dataset, num_images=3):
    plt.figure(figsize=(15, 5))
    for i, features in enumerate(dataset.take(num_images)):
        for j in range(features.shape[-1]):
            plt.subplot(num_images, features.shape[-1], i * features.shape[-1] + j + 1)
            plt.imshow(features[0, :, :, j], cmap='gray')
            plt.title(f'Feature {INPUT_FEATURES[j]}')
            plt.axis('off')
    plt.show()

# 显示前3个图像
display_images(dataset, num_images=3)
