import time

import cv2
import numpy as np
import tensorflow as tf
from termcolor import cprint
import tensorflow as tf
import warnings
import logging
from SubFunctions.EdgeAware import sobelLayer
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_CHANNELS = 1



class Downsample(tf.keras.Model):
    def __init__(self, filters, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()



    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Upsample(tf.keras.Model):
    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.up_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                       (size, size),
                                                       strides=2,
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x1, x2, training):
        x = self.up_conv(x1)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, x2], axis=-1)
        return x


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = Downsample(64, 4, apply_batchnorm=False)
        self.down2 = Downsample(128, 4)
        self.down3 = Downsample(256, 4)
        self.down4 = Downsample(512, 4)
        self.down5 = Downsample(512, 4)
        self.down6 = Downsample(512, 4)
        self.down7 = Downsample(512, 4)
        self.down8 = Downsample(512, 4)

        self.up1 = Upsample(512, 4, apply_dropout=True)
        self.up2 = Upsample(512, 4, apply_dropout=True)
        self.up3 = Upsample(512, 4, apply_dropout=True)
        self.up4 = Upsample(512, 4)
        self.up5 = Upsample(256, 4)
        self.up6 = Upsample(128, 4)
        self.up7 = Upsample(64, 4)

        self.last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS,
                                                    (4, 4),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer)

    def call(self, x, training):
        # x shape == (bs, 256, 256, 3)
        x1 = self.down1(x, training=training)  # (bs, 128, 128, 64)
        x2 = self.down2(x1, training=training)  # (bs, 64, 64, 128)
        x3 = self.down3(x2, training=training)  # (bs, 32, 32, 256)
        x4 = self.down4(x3, training=training)  # (bs, 16, 16, 512)
        x5 = self.down5(x4, training=training)  # (bs, 8, 8, 512)
        x6 = self.down6(x5, training=training)  # (bs, 4, 4, 512)
        x7 = self.down7(x6, training=training)  # (bs, 2, 2, 512)
        x8 = self.down8(x7, training=training)  # (bs, 1, 1, 512)


        x9 = self.up1(x8, x7, training=training)  # (bs, 2, 2, 1024)
        x10 = self.up2(x9, x6, training=training)  # (bs, 4, 4, 1024)
        x11 = self.up3(x10, x5, training=training)  # (bs, 8, 8, 1024)
        x12 = self.up4(x11, x4, training=training)  # (bs, 16, 16, 1024)
        x13 = self.up5(x12, x3, training=training)  # (bs, 32, 32, 512)
        x14 = self.up6(x13, x2, training=training)  # (bs, 64, 64, 256)
        x15 = self.up7(x14, x1, training=training)  # (bs, 128, 128, 128)

        x16 = self.last(x15)  # (bs, 256, 256, 3)
        x16 = tf.nn.tanh(x16)

        return x16


class DiscDownsample(tf.keras.Model):
    def __init__(self, filters, size, apply_batchnorm=True):
        super(DiscDownsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = DiscDownsample(64, 4, False)
        self.down2 = DiscDownsample(128, 4)
        self.down3 = DiscDownsample(256, 4)

        # we are zero padding here with 1 because we need our shape to
        # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
        self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(512,
                                           (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer,
                                           use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.last = tf.keras.layers.Conv2D(1,
                                           (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer)

    def call(self, inp, tar, training):
        # concatenating the input and the target
        x = tf.concat([inp, tar], axis=-1)  # (bs, 256, 256, channels*2)
        x = self.down1(x, training=training)  # (bs, 128, 128, 64)
        x = self.down2(x, training=training)  # (bs, 64, 64, 128)
        x = self.down3(x, training=training)  # (bs, 32, 32, 256)

        x = self.zero_pad1(x)  # (bs, 34, 34, 256)
        x = self.conv(x)  # (bs, 31, 31, 512)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.zero_pad2(x)  # (bs, 33, 33, 512)
        # don't add a sigmoid activation here since
        # the loss function expects raw logits.
        x = self.last(x)  # (bs, cen, 30, 1 )

        return x


class SegmentationGAN:
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.generator = Generator()
            self.discriminator = Discriminator()

        self.LAMBDA = 100
        self.generator_optimizer = tf.optimizers.Adam(beta_1=0.5, epsilon=2e-4)
        self.discriminator_optimizer = tf.optimizers.Adam(beta_1=0.5, epsilon=2e-4)

    @staticmethod
    def discriminator_loss(disc_real_output, disc_generated_output):
        real_loss = tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_real_output),
                                                              logits=disc_real_output)
        generated_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.zeros_like(disc_generated_output), logits=disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss


    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_generated_output),
                                                             logits=disc_generated_output)
        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss

    def generate_images(self, model, test_input, tar, save, address=None):
        global name

        prediction = model(test_input, training=True)
        prediction = self.mask
        prediction[prediction != 0] = 1
        region_mask = (prediction == 1)

        # Create empty canvas
        region = np.zeros_like(self.image)
        region[region_mask] = self.image[region_mask]

        return region






    def get_segmented(self, epochs=5):

        dataset = [[self.image, self.mask]]


        for epoch in range(epochs):
            start = time.time()
            epoch_gen_loss, epoch_disc_loss = [], []

            for input_image, target in dataset:
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    input_image = cv2.resize(input_image, (256, 256))
                    input_image = np.expand_dims(input_image, axis=0)
                    input_image = np.expand_dims(input_image, axis=-1)
                    gen_output = self.generator(input_image, training=True)
                    target[target != 0] = 1
                    target = cv2.resize(target, (256, 256))
                    target = np.expand_dims(target, axis=0)
                    target = np.expand_dims(target, axis=-1)

                    # Edge aware
                    edge_aware_target = sobelLayer(np.squeeze(target))
                    edge_aware_gen_output = sobelLayer(np.squeeze(gen_output))
                    target += edge_aware_target
                    gen_output += edge_aware_gen_output


                    disc_real_output = self.discriminator(input_image, target, training=True)
                    disc_generated_output = self.discriminator(input_image, gen_output, training=True)

                    gen_loss = self.generator_loss(disc_generated_output, gen_output, target)
                    disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
                    # losses.append(zip(gen_loss, disc_loss))
                    epoch_gen_loss.append(gen_loss.numpy())
                    epoch_disc_loss.append(disc_loss.numpy())

                generator_gradients = gen_tape.gradient(gen_loss, self.generator.variables)
                discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.variables)

                self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                        self.generator.variables))
                self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                            self.discriminator.variables))
            elapsed = time.time() - start
            # Epoch summary
            avg_gen_loss = np.mean(epoch_gen_loss)
            avg_disc_loss = np.mean(epoch_disc_loss)
            cprint("=" * 50, "cyan")
            cprint(f"📌 Epoch [{epoch + 1}/{epochs}] Completed", "yellow", attrs=["bold"])
            cprint(f"   ⏱ Time Taken: {elapsed:.2f} sec", "green")
            cprint(f"   🎨 Generator Loss: {avg_gen_loss:.4f}", "blue")
            cprint(f"   🛡 Discriminator Loss: {avg_disc_loss:.4f}", "magenta")
            cprint("=" * 50 + "\n", "cyan")

            output = self.generate_images(self.generator, input_image, target, save=False)

            return output



