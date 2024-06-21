import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LayerNormalization, Layer, Input
from tensorflow.keras.models import Model
from preprocess import INPUT_FEATURES

class SwinTransformerBlock(Layer):
    def __init__(self, num_heads, window_size, shift_size, **kwargs):
        super(SwinTransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.layer_norm = LayerNormalization()
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=window_size)

    def call(self, inputs):
        x = self.layer_norm(inputs)
        x = self.multi_head_attention(x, x)
        if self.shift_size > 0:
            x = tf.roll(x, shift=self.shift_size, axis=1)
        return x

def SwinUNet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = SwinTransformerBlock(num_heads=4, window_size=4, shift_size=2)(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

input_shape = (32, 32, len(INPUT_FEATURES))
num_classes = 1
model = SwinUNet(input_shape, num_classes)
model.summary()
