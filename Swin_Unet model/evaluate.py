import tensorflow as tf
from preprocess import get_dataset, INPUT_FEATURES

test_file_pattern = '/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/test/*_ongoing_*.tfrecord'
data_size = 64
sample_size = 32
batch_size = 16
num_in_channels = len(INPUT_FEATURES)
compression_type = ''  # Set to 'GZIP' if files are gzip compressed
clip_and_normalize = True
clip_and_rescale = False
random_crop = False
center_crop = True

test_dataset = get_dataset(
    test_file_pattern,
    data_size,
    sample_size,
    batch_size,
    num_in_channels,
    compression_type,
    clip_and_normalize,
    clip_and_rescale,
    random_crop,
    center_crop,
)

# Load the model
model = tf.keras.models.load_model('swin_unet_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}')
