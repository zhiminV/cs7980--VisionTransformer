import matplotlib.pyplot as plt
from matplotlib import colors
import tensorflow as tf
from preprocess_data import get_dataset

# Load the datasets
BATCH_SIZE = 32

train_dataset = get_dataset(
    '/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/train/*_ongoing_*.tfrecord', 
    data_size=64, 
    sample_size=64, 
    batch_size=BATCH_SIZE, 
    num_in_channels=12, 
    compression_type=None, 
    clip_and_normalize=True, 
    clip_and_rescale=False, 
    random_crop=True, 
    center_crop=False
)

validation_dataset = get_dataset(
    '/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/val/*_ongoing_*.tfrecord', 
    data_size=64, 
    sample_size=64, 
    batch_size=BATCH_SIZE, 
    num_in_channels=12, 
    compression_type=None, 
    clip_and_normalize=True, 
    clip_and_rescale=False, 
    random_crop=True, 
    center_crop=False
)

test_dataset = get_dataset(
    '/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/test/*_ongoing_*.tfrecord', 
    data_size=64, 
    sample_size=64, 
    batch_size=BATCH_SIZE, 
    num_in_channels=12, 
    compression_type=None, 
    clip_and_normalize=True, 
    clip_and_rescale=False, 
    random_crop=True, 
    center_crop=False
)

# Titles for the features
TITLES = [
    'Elevation',
    'WindDirection',
    'WindVelocity',
    'MinTemp',
    'MaxTemp',
    'Humidity',
    'Precipitation',
    'Drought',
    'Vegetation',
    'PopulationDensity',
    'ERC',
    'PreFireMask',
    'FireMask'
]

def plot_sample_from_dataset(dataset: tf.data.Dataset):
    """
    Plot one row of samples from the dataset showing 12 features and fire mask.

    Args:
        dataset (tf.data.Dataset): Dataset from which to plot samples.
    """
    global TITLES

    # Get a batch
    inputs, labels = None, None
    for elem in dataset:
        inputs, labels = elem
        break

    # Select the first sample
    sample_inputs = inputs[0]
    sample_label = labels[0]

    fig, axs = plt.subplots(1, 13, figsize=(25, 5))

    # Variables for controlling the color map for the fire masks
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)

    for j in range(12):
        ax = axs[j]
        ax.imshow(sample_inputs[:, :, j], cmap='viridis')
        ax.set_title(TITLES[j], fontsize=13)
        ax.axis('off')
        # Add height and width annotations
        ax.text(0.5, -0.1, f'{sample_inputs.shape[0]}x{sample_inputs.shape[1]}', size=12, ha='center', transform=ax.transAxes)

    # Plot the fire mask
    ax = axs[12]
    ax.imshow(sample_label[:, :, 0], cmap=CMAP, norm=NORM)
    ax.set_title(TITLES[12], fontsize=13)
    ax.axis('off')
    # Add height and width annotations
    ax.text(0.5, -0.1, f'{sample_label.shape[0]}x{sample_label.shape[1]}', size=12, ha='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()

# Example usage
plot_sample_from_dataset(train_dataset)
