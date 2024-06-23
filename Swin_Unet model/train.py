import tensorflow as tf
from preprocess import get_dataset, INPUT_FEATURES
from swin_unet import SwinUNet  # Ensure this import matches your actual Swin UNet implementation

train_file_pattern = '/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/train/*_ongoing_*.tfrecord'
val_file_pattern = '/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/val/*_ongoing_*.tfrecord'
test_file_pattern = '/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/test/*_ongoing_*.tfrecord'
data_size = 64
sample_size = 32
batch_size = 16
num_in_channels = len(INPUT_FEATURES)
compression_type = ''  # Set to 'GZIP' if your files are gzip compressed
clip_and_normalize = True
clip_and_rescale = False
random_crop = False
center_crop = True

train_dataset = get_dataset(
    train_file_pattern,
    data_size,
    sample_size,
    batch_size,
    num_in_channels,
    compression_type,
    clip_and_normalize,
    clip_and_rescale,
    random_crop,
    center_crop,
).repeat()

val_dataset = get_dataset(
    val_file_pattern,
    data_size,
    sample_size,
    batch_size,
    num_in_channels,
    compression_type,
    clip_and_normalize,
    clip_and_rescale,
    random_crop,
    center_crop,
).repeat()

model = SwinUNet((sample_size, sample_size, num_in_channels), 1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# Save the model
model.save('swin_unet_model.h5')

# Plot accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
