import tensorflow as tf
from tensorflow.keras import layers as kl


def _upsample_like(src, tar):
    tar_h = tf.shape(tar)[1]
    tar_w = tf.shape(tar)[2]
    src = tf.image.resize(src, size=(tar_h, tar_w), method='bilinear')
    return src


class REBNCONV(tf.keras.layers.Layer):
    def __init__(self, out_ch=3, dirate=1, name=None):
        super(REBNCONV, self).__init__(name=name)
        self.conv_s1 = kl.Conv2D(filters=out_ch, kernel_size=3, padding='same', dilation_rate=dirate)
        self.bn_s1 = kl.BatchNormalization()

    def call(self, x, training=False):
        x = self.conv_s1(x)
        x = self.bn_s1(x, training)
        x = tf.nn.relu(x)
        return x


### RSU-7 ###
class RSU7(tf.keras.layers.Layer):
    def __init__(self, mid_ch=12, out_ch=3, name=None):
        super(RSU7, self).__init__(name=name)
        self.rebnconvin = REBNCONV(out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(mid_ch, dirate=1)
        self.pool1 = kl.MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.rebnconv2 = REBNCONV(mid_ch, dirate=1)
        self.pool2 = kl.MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.rebnconv3 = REBNCONV(mid_ch, dirate=1)
        self.pool3 = kl.MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.rebnconv4 = REBNCONV(mid_ch, dirate=1)
        self.pool4 = kl.MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.rebnconv5 = REBNCONV(mid_ch, dirate=1)
        self.pool5 = kl.MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.rebnconv6 = REBNCONV(mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(out_ch, dirate=1)

    def call(self, x, training=None):
        hx = x
        hxin = self.rebnconvin(hx, training)

        hx1 = self.rebnconv1(hxin, training)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx, training)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx, training)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx, training)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx, training)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx, training)

        hx7 = self.rebnconv7(hx6, training)

        hx6d = self.rebnconv6d(tf.concat([hx7, hx6], 3), training)  # okay
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(tf.concat([hx6dup, hx5], 3), training)
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(tf.concat([hx5dup, hx4], 3), training)
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(tf.concat([hx4dup, hx3], 3), training)
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(tf.concat([hx3dup, hx2], 3), training)
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(tf.concat([hx2dup, hx1], 3), training)

        return hx1d + hxin  # hx1d and hxin --> shapes are different


### RSU-6 ###
class RSU6(tf.keras.layers.Layer):
    def __init__(self, mid_ch=12, out_ch=3, name=None):
        super(RSU6, self).__init__(name=name)

        self.rebnconvin = REBNCONV(out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(mid_ch, dirate=1)
        self.pool1 = kl.MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.rebnconv2 = REBNCONV(mid_ch, dirate=1)
        self.pool2 = kl.MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.rebnconv3 = REBNCONV(mid_ch, dirate=1)
        self.pool3 = kl.MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.rebnconv4 = REBNCONV(mid_ch, dirate=1)
        self.pool4 = kl.MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.rebnconv5 = REBNCONV(mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(out_ch, dirate=1)

    def call(self, x, training=None):
        hx = x

        hxin = self.rebnconvin(hx, training)

        hx1 = self.rebnconv1(hxin, training)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx, training)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx, training)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx, training)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx, training)

        hx6 = self.rebnconv6(hx5, training)

        hx5d = self.rebnconv5d(tf.concat([hx6, hx5], 3), training)
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(tf.concat([hx5dup, hx4], 3), training)
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(tf.concat([hx4dup, hx3], 3), training)
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(tf.concat([hx3dup, hx2], 3), training)
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(tf.concat([hx2dup, hx1], 3), training)

        return hx1d + hxin


### RSU-5 ###
class RSU5(tf.keras.layers.Layer):

    def __init__(self, mid_ch=12, out_ch=3, name=None):
        super(RSU5, self).__init__(name=name)

        self.rebnconvin = REBNCONV(out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(mid_ch, dirate=1)
        self.pool1 = kl.MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.rebnconv2 = REBNCONV(mid_ch, dirate=1)
        self.pool2 = kl.MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.rebnconv3 = REBNCONV(mid_ch, dirate=1)
        self.pool3 = kl.MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.rebnconv4 = REBNCONV(mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(out_ch, dirate=1)

    def call(self, x, training=None):
        hx = x

        hxin = self.rebnconvin(hx, training)

        hx1 = self.rebnconv1(hxin, training)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx, training)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx, training)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx, training)

        hx5 = self.rebnconv5(hx4, training)

        hx4d = self.rebnconv4d(tf.concat([hx5, hx4], 3), training)
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(tf.concat([hx4dup, hx3], 3), training)
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(tf.concat([hx3dup, hx2], 3), training)
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(tf.concat([hx2dup, hx1], 3), training)

        return hx1d + hxin


### RSU-4 ###
class RSU4(tf.keras.layers.Layer):

    def __init__(self, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(mid_ch, dirate=1)
        self.pool1 = kl.MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.rebnconv2 = REBNCONV(mid_ch, dirate=1)
        self.pool2 = kl.MaxPool2D(pool_size=2, strides=2, padding='valid')

        self.rebnconv3 = REBNCONV(mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(out_ch, dirate=1)

    def call(self, x, training=None):
        hx = x

        hxin = self.rebnconvin(hx, training)

        hx1 = self.rebnconv1(hxin, training)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx, training)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx, training)

        hx4 = self.rebnconv4(hx3, training)

        hx3d = self.rebnconv3d(tf.concat([hx4, hx3], 3), training)
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(tf.concat([hx3dup, hx2], 3), training)
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(tf.concat([hx2dup, hx1], 3), training)

        return hx1d + hxin


### RSU-4F ###
class RSU4F(tf.keras.layers.Layer):

    def __init__(self, mid_ch=12, out_ch=3, name=None):
        super(RSU4F, self).__init__(name=name)

        self.rebnconvin = REBNCONV(out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(out_ch, dirate=1)

    def call(self, x, training=None):
        hx = x

        hxin = self.rebnconvin(hx, training)

        hx1 = self.rebnconv1(hxin, training)
        hx2 = self.rebnconv2(hx1, training)
        hx3 = self.rebnconv3(hx2, training)

        hx4 = self.rebnconv4(hx3, training)

        hx3d = self.rebnconv3d(tf.concat([hx4, hx3], 3), training)
        hx2d = self.rebnconv2d(tf.concat([hx3d, hx2], 3), training)
        hx1d = self.rebnconv1d(tf.concat([hx2d, hx1], 3), training)

        return hx1d + hxin


def U2Net(input_shape):
    inputs = kl.Input(input_shape)

    # Encoder
    x1 = RSU7(32, 64)(inputs)
    x = kl.MaxPool2D(2, 2)(x1)

    x2 = RSU6(32, 128)(x)
    x = kl.MaxPool2D(2, 2)(x2)

    x3 = RSU5(64, 256)(x)
    x = kl.MaxPool2D(2, 2)(x3)

    x4 = RSU4(128, 512)(x)
    x = kl.MaxPool2D(2, 2)(x4)

    x5 = RSU4F(256, 512)(x)
    x = kl.MaxPool2D(2, 2)(x5)

    x6 = RSU4F(256, 512)(x)
    x = kl.UpSampling2D(size=[2, 2], interpolation="bilinear")(x6)

    # Decoder
    x = kl.Concatenate()([x, x5])
    x5d = RSU4F(256, 512)(x)
    x = kl.UpSampling2D(size=[2, 2], interpolation="bilinear")(x5d)

    x = kl.Concatenate()([x, x4])
    x4d = RSU4(128, 256)(x)
    x = kl.UpSampling2D(size=[2, 2], interpolation="bilinear")(x4d)

    x = kl.Concatenate()([x, x3])
    x3d = RSU5(64, 128)(x)
    x = kl.UpSampling2D(size=[2, 2], interpolation="bilinear")(x3d)

    x = kl.Concatenate()([x, x2])
    x2d = RSU6(32, 64)(x)
    x = kl.UpSampling2D(size=[2, 2], interpolation="bilinear")(x2d)

    x = kl.Concatenate()([x, x1])
    x1d = RSU7(16, 64)(x)

    # Side outputs
    s1 = kl.Conv2D(1, 3, padding="same")(x1d)

    s2 = kl.Conv2D(1, 3, padding="same")(x2d)
    s2 = kl.UpSampling2D(size=[2, 2], interpolation="bilinear")(s2)

    s3 = kl.Conv2D(1, 3, padding="same")(x3d)
    s3 = kl.UpSampling2D(size=[4, 4], interpolation="bilinear")(s3)

    s4 = kl.Conv2D(1, 3, padding="same")(x4d)
    s4 = kl.UpSampling2D(size=[8, 8], interpolation="bilinear")(s4)

    s5 = kl.Conv2D(1, 3, padding="same")(x5d)
    s5 = kl.UpSampling2D(size=[16, 16], interpolation="bilinear")(s5)

    s6 = kl.Conv2D(1, 3, padding="same")(x6)
    s6 = kl.UpSampling2D(size=[32, 32], interpolation="bilinear")(s6)

    s0 = kl.Concatenate()([s1, s2, s3, s4, s5, s6])
    s0 = kl.Conv2D(1, 1, padding="same")(s0)

    s0 = kl.Activation("sigmoid", name="s0")(s0)
    s1 = kl.Activation("sigmoid", name="s1")(s1)
    s2 = kl.Activation("sigmoid", name="s2")(s2)
    s3 = kl.Activation("sigmoid", name="s3")(s3)
    s4 = kl.Activation("sigmoid", name="s4")(s4)
    s5 = kl.Activation("sigmoid", name="s5")(s5)
    s6 = kl.Activation("sigmoid", name="s6")(s6)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[s0, s1, s2, s3, s4, s5, s6], name="U2Net")
    return model
