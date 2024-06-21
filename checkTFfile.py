import tensorflow as tf

def inspect_tfrecord(file_pattern):
    raw_dataset = tf.data.TFRecordDataset(file_pattern)
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)

train_file_pattern = '/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/train/*_ongoing_*.tfrecord'
inspect_tfrecord(train_file_pattern)
