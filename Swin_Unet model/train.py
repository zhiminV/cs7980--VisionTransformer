import tensorflow as tf
from preprocess import get_dataset, INPUT_FEATURES
from model import SwinUNet

train_file_pattern = '/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/train/*_ongoing_*.tfrecord'
val_file_pattern = '/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/val/*_ongoing_*.tfrecord'
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
) # can add .repeat()
# warning：         [[{{node IteratorGetNext}}]]
#/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/contextlib.py:153: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  ##self.gen.throw(typ, value, traceback)
#2024-06-19 23:44:47.220510: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence.repeat() 

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
)

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
