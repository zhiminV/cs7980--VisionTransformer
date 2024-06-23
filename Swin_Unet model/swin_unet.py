import tensorflow as tf
from tensorflow.keras import layers

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = tf.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    windows = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(windows, (-1, window_size, window_size, C))
    return windows

def window_reverse(windows, window_size, H, W, C):
    B = tf.shape(windows)[0] // (H * W // window_size // window_size)
    x = tf.reshape(windows, (B, H // window_size, W // window_size, window_size, window_size, C))
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, (B, H, W, C))
    return x

class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(dropout)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(dropout)
        
    def call(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B_, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))
        
        if mask is not None:
            nW = mask.shape[0]
            attn = tf.reshape(attn, (B_ // nW, nW, self.num_heads, N, N)) + mask
            attn = tf.reshape(attn, (B_, self.num_heads, N, N))
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)
        
        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B_, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias=qkv_bias, dropout=dropout)
        
        self.drop_path = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential([
            layers.Dense(int(dim * mlp_ratio)),
            layers.Activation('gelu'),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout)
        ])
        
        if min(self.window_size) <= 0:
            self.shift_size = 0
            self.window_size = min(self.window_size)
        
    def build(self, input_shape):
        H, W = input_shape[1], input_shape[2]
        if min(H, W) < self.window_size:
            self.window_size = min(H, W)
        
        if self.shift_size > 0:
            img_mask = np.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(tf.convert_to_tensor(img_mask), self.window_size)
            mask_windows = tf.reshape(mask_windows, (-1, self.window_size * self.window_size))
            attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
            attn_mask = tf.where(attn_mask != 0, float(-100.0), attn_mask)
            attn_mask = tf.where(attn_mask == 0, float(0.0), attn_mask)
        else:
            attn_mask = None
            
        self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)
        
    def call(self, x):
        H, W = self.input_shape[1], self.input_shape[2]
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, (-1, H, W, C))
        
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x
        
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, (-1, self.window_size * self.window_size, C))
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        attn_windows = tf.reshape(attn_windows, (-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)
        
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        
        x = tf.reshape(x, (-1, H * W, C))
        x = self.drop_path(x) + shortcut
        x = self.drop_path(self.mlp(self.norm2(x))) + x
        return x
def SwinUNet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = SwinTransformerBlock(dim=64, num_heads=4, window_size=4, shift_size=2)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = SwinTransformerBlock(dim=64, num_heads=4, window_size=4, shift_size=2)(x)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs, name="SwinUNet")
    return model
