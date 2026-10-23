import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------ BasicConv ------------------

class AFF(tf.keras.layers.Layer):
    """
    Adaptive Feature Fusion (AFF)
    """

    def __init__(self, channels=64, r=4, **kwargs):
        super(AFF, self).__init__(**kwargs)
        inter_channels = channels // r

        # Local attention
        self.local_att = models.Sequential([
            layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(channels, kernel_size=1, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization()
        ])

        # Global attention
        self.global_avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.global_att = models.Sequential([
            layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(channels, kernel_size=1, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization()
        ])

        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x, residual):
        xa = x + residual

        # Local attention
        xl = self.local_att(xa)

        # Global attention
        xg_pool = self.global_avg_pool(xa)
        xg = self.global_att(xg_pool)

        # Combine
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo



import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------------------------------------
# Channel Squeeze & Excitation (cSE)
# ---------------------------------------------------------
class ChannelSELayer(tf.keras.layers.Layer):
    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer, self).__init__()
        reduced_channels = num_channels // reduction_ratio

        self.global_avg = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(reduced_channels, activation='relu')
        self.fc2 = layers.Dense(num_channels, activation='sigmoid')

    def call(self, x):
        # (B, H, W, C) → (B, C)
        squeeze = self.global_avg(x)

        # Fully connected bottleneck
        excitation = self.fc2(self.fc1(squeeze))  # (B, C)
        excitation = tf.reshape(excitation, (-1, 1, 1, x.shape[-1]))
        excitation = tf.reshape(excitation, (-1, 1, 1, x.shape[-1]))

        return x * excitation


# ---------------------------------------------------------
# Spatial Squeeze & Excitation (sSE)
# ---------------------------------------------------------
class SpatialSELayer(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(SpatialSELayer, self).__init__()
        self.conv = layers.Conv2D(1, kernel_size=1, activation='sigmoid')

    def call(self, x):
        # Spatial attention → (B, H, W, 1)
        squeeze = self.conv(x)
        return x * squeeze




# Spatial Attention Module (SAM)
class SAM(tf.keras.layers.Layer):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=7,
            strides=1,
            padding='same',
            use_bias=bias
        )

    def call(self, x):
        # Compute max and mean along the channel axis
        max_pool = tf.reduce_max(x, axis=3, keepdims=True)
        avg_pool = tf.reduce_mean(x, axis=3, keepdims=True)
        concat = tf.concat([max_pool, avg_pool], axis=3)

        # Convolution + sigmoid
        output = tf.nn.sigmoid(self.conv(concat)) * x
        return output


# Channel Attention Module (CAM)
class CAM(tf.keras.layers.Layer):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r

        self.mlp = tf.keras.Sequential([
            layers.Dense(channels // r, activation='relu'),
            layers.Dense(channels)
        ])

    def call(self, x):
        # Adaptive pooling to (1,1)
        max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

        # Flatten for fully connected layers
        max_pool_flat = tf.reshape(max_pool, [tf.shape(x)[0], self.channels])
        avg_pool_flat = tf.reshape(avg_pool, [tf.shape(x)[0], self.channels])

        # MLP and reshape back
        max_out = tf.reshape(self.mlp(max_pool_flat), [-1, 1, 1, self.channels])
        avg_out = tf.reshape(self.mlp(avg_pool_flat), [-1, 1, 1, self.channels])

        # Combine and apply sigmoid
        out = tf.nn.sigmoid(max_out + avg_out) * x
        return out


# ---------------------------------------------------------
# Concurrent Channel & Spatial SE (scSE)
# ---------------------------------------------------------
class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, num_channels, reduction_ratio=2):
        super(SqueezeExcitation, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def call(self, x):
        cse_out = self.cSE(x)
        sse_out = self.sSE(x)
        return tf.maximum(cse_out, sse_out)  # elementwise max


# CBAM Block (Channel + Spatial Attention)
class CBAM(tf.keras.layers.Layer):
    def __init__(self, channels, r=16):
        super(CBAM, self).__init__()
        self.cam = CAM(channels, r)
        self.sam = SAM(bias=False)

    def call(self, x):
        out = self.cam(x)
        out = self.sam(out)
        return out + x


class EfficientChannel(tf.keras.layers.Layer):
    """
    Keras implementation of Efficient Channel Attention (ECA)
    Equivalent to the PyTorch version.
    """

    def __init__(self, channels, k_size=3):
        super(EfficientChannel, self).__init__()
        self.global_avg = layers.GlobalAveragePooling2D(keepdims=True)

        # Conv1D over channels → use DepthwiseConv2D with H=1, W=k_size
        self.conv = layers.Conv1D(
            filters=1,
            kernel_size=k_size,
            padding="same",
            use_bias=False
        )

    def call(self, x):
        # x: (B, H, W, C)
        # Global Avg Pool → (B, 1, 1, C)
        y = self.global_avg(x)                # (B, 1, 1, C)

        # Flatten height & width → 1D conv on Channels
        y = tf.squeeze(y, axis=1)            # (B, 1, C)
        y = self.conv(y)                     # (B, 1, C)
        y = tf.expand_dims(y, axis=1)         # (B, 1, 1, C)

        # Sigmoid activation
        y = tf.sigmoid(y)

        # Apply attention
        return x * y



class SpatialGroupEnhance(tf.keras.layers.Layer):
    """
    Keras implementation of Spatial Group Enhance (SGE)
    Equivalent to the provided PyTorch version.
    """

    def __init__(self, groups):
        super(SpatialGroupEnhance, self).__init__()
        self.groups = groups
        print("add one SGE!")

    def build(self, input_shape):
        channels = input_shape[-1]
        assert channels % self.groups == 0, "Channels must be divisible by groups."

        # learnable parameters (1, groups, 1, 1)
        self.weight = self.add_weight(
            shape=(1, self.groups, 1, 1),
            initializer=tf.zeros_initializer(),
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(1, self.groups, 1, 1),
            initializer=tf.ones_initializer(),
            trainable=True,
        )

        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)

    def call(self, x):
        # Input: (B, H, W, C)
        b, h, w, c = tf.unstack(tf.shape(x))
        group_channels = c // self.groups

        # reshape to (B, H, W, groups, group_channels)
        x_group = tf.reshape(x, (b, h, w, self.groups, group_channels))
        x_group = tf.transpose(x_group, [0, 3, 4, 1, 2])  # (B, groups, C', H, W)

        # merge batch and group dims → (B*groups, C', H, W)
        x_reshape = tf.reshape(x_group, (b * self.groups, group_channels, h, w))

        # avg pool (per sample)
        avg = self.avg_pool(tf.transpose(x_reshape, [0, 2, 3, 1]))  # (b*g, 1, 1, C')
        avg = tf.transpose(avg, [0, 3, 1, 2])                      # (b*g, C', 1, 1)

        # multiply x * avg
        xn = x_reshape * avg

        # sum along channels → (b*g, 1, H, W)
        xn_sum = tf.reduce_sum(xn, axis=1, keepdims=True)

        # normalize
        t = tf.reshape(xn_sum, (b * self.groups, -1))            # (b*g, H*W)
        t_mean = tf.reduce_mean(t, axis=1, keepdims=True)
        t = t - t_mean

        t_std = tf.math.reduce_std(t, axis=1, keepdims=True) + 1e-5
        t = t / t_std

        # reshape back to (B, groups, H, W)
        t = tf.reshape(t, (b, self.groups, h, w))

        # t = t * weight + bias
        t = t * self.weight + self.bias

        # reshape to (b*groups, 1, h, w)
        t = tf.reshape(t, (b * self.groups, 1, h, w))
        t = tf.sigmoid(t)

        # apply the attention
        out = x_reshape * t

        # reshape back to original shape
        out = tf.reshape(out, (b, self.groups, group_channels, h, w))
        out = tf.transpose(out, [0, 3, 4, 1, 2])  # (b, h, w, groups, C')
        out = tf.reshape(out, (b, h, w, c))

        return out
