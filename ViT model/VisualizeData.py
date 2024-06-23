
import matplotlib.pyplot as plt
from matplotlib import colors
import tensorflow as tf
from preprocess_data import get_dataset

#Load the datasets
BATCH_SIZE = 32

train_dataset = get_dataset('/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/train/*_ongoing_*.tfrecord', data_size=64, sample_size=32, batch_size=BATCH_SIZE, num_in_channels=12, compression_type=None, clip_and_normalize=True, clip_and_rescale=False, random_crop=True, center_crop=False)

validation_dataset = get_dataset('/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/val/*_ongoing_*.tfrecord', data_size=64, sample_size=32, batch_size=BATCH_SIZE, num_in_channels=12, compression_type=None, clip_and_normalize=True, clip_and_rescale=False, random_crop=True, center_crop=False)

test_dataset = get_dataset('/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/test/*_ongoing_*.tfrecord', data_size=64, sample_size=32, batch_size=BATCH_SIZE, num_in_channels=12, compression_type=None, clip_and_normalize=True, clip_and_rescale=False, random_crop=True, center_crop=False)

# visualize data:
# TITLES = [
#   'Elevation',
#   'Wind\ndirection',
#   'Wind\nvelocity',
#   'Min\ntemp',
#   'Max\ntemp',
#   'Humidity',
#   'Precip',
#   'Drought',
#   'Vegetation',
#   'Population\ndensity',
#   'Energy\nrelease\ncomponent',
#   'Previous\nfire\nmask',
#   'Fire\nmask'
# ]
TITLES = [
    'elevation',
    'th',
    'sph',
    'pr',
    'NDVI',
    'PrevFireMask',
]

def plot_samples_from_dataset(dataset: tf.data.Dataset, n_rows: int):
    """
    Plot 'n_rows' rows of samples from dataset.

    Args:
        dataset (Dataset): Dataset from which to plot samples.
        n_rows (int): Number of rows to plot.
    """
    global TITLES

    # Get batch
    inputs, labels = None, None
    for elem in dataset:
        inputs, labels = elem
        break

    fig = plt.figure(figsize=(15,6.5))

    # Variables for controllong the color map for the fire masks
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)
    # Number of data variables
    n_features = 5
    for i in range(n_rows):
        for j in range(n_features + 1):
            plt.subplot(n_rows, n_features + 1, i * (n_features + 1) + j + 1)
            if i == 0:
                plt.title(TITLES[j], fontsize=13)
            if j < n_features - 1:
                plt.imshow(inputs[i, :, :, j], cmap='viridis')
            if j == n_features - 1:
                plt.imshow(inputs[i, :, :, -1], cmap=CMAP, norm=NORM)
            if j == n_features:
                plt.imshow(labels[i, :, :, 0], cmap=CMAP, norm=NORM)
            plt.axis('off')
    plt.tight_layout()
    plt.show()
    

# Example usage
plot_samples_from_dataset(train_dataset, 5)