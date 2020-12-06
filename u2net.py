import tensorflow as tf
from tensorflow.keras import layers as kl


class REBNCONV(tf.keras.layers.Layer):
    def __init__(self, filters=3, dilation_rate=1, name=None):
        super(REBNCONV, self).__init__(name=name)
        self.filters = filters
        self.dilation_rate = dilation_rate

        self.conv_s1 = kl.Conv2D(filters=self.filters, kernel_size=3, padding='same', dilation_rate=self.dilation_rate)
        self.bn_s1 = kl.BatchNormalization()

    def call(self, x, training=False):
        x = self.conv_s1(x)
        x = self.bn_s1(x, training)
        x = tf.nn.relu(x)
        return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'dilation_rate': self.dilation_rate,
        }
        base_config = super(REBNCONV, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RSU_N(tf.keras.layers.Layer):
    _ids = {}

    def __init__(self, n, filters=12, filters_out=3, name=None):
        if name is None:
            name_rsu = "RSU{}".format(n)
            id_rsu = RSU_N._ids.setdefault(name_rsu, 1)
            name = name_rsu + "_" + str(id_rsu)
            RSU_N._ids[name_rsu] += 1

        super(RSU_N, self).__init__(name=name)
        self.n = n
        self.filters = filters
        self.filters_out = filters_out

        self.conv_in = REBNCONV(self.filters_out, dilation_rate=1)
        self.conv_encoder = [REBNCONV(self.filters, dilation_rate=1) for _ in range(self.n - 1)]
        self.max_pools = [kl.MaxPool2D(pool_size=2, strides=2, padding='valid') for _ in range(self.n - 2)]

        self.conv_n = REBNCONV(self.filters, dilation_rate=2)

        self.conv_decoder = [REBNCONV(self.filters, dilation_rate=1) for _ in range(self.n - 1)]
        self.upsamples = [kl.UpSampling2D(size=[2, 2], interpolation='bilinear') for _ in range(self.n - 2)]
        self.conv_out = REBNCONV(self.filters_out, dilation_rate=1)

    def call(self, inputs, training=None):
        xin = self.conv_in(inputs, training=training)

        # encoder
        x = xin
        x_is = []
        for i in range(self.n - 2):
            x_i = self.conv_encoder[i](x, training=training)
            x = self.max_pools[i](x_i)
            x_is.append(x_i)
        x_is.append(self.conv_encoder[-1](x, training=training))

        # mid
        x = self.conv_n(x_is[-1], training=training)

        # decoder
        for i in range(self.n - 3, -1, -1):
            x_i = x_is[i + 1]
            x = kl.Concatenate()([x, x_i])
            xid = self.conv_decoder[i](x, training=training)
            x = self.upsamples[i](xid)

        x = kl.Concatenate()([x, x_is[0]])
        x = self.conv_out(x, training=training)
        return xin + x

    def get_config(self):
        config = {
            'n': self.n,
            'filters': self.filters,
            'filters_out': self.filters_out,
        }
        base_config = super(RSU_N, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RSU_NF(tf.keras.layers.Layer):
    _ids = {}

    def __init__(self, n, filters=12, filters_out=3, name=None):
        if name is None:
            name_rsu = "RSU{}F".format(n)
            id_rsu = RSU_NF._ids.setdefault(name_rsu, 1)
            name = name_rsu + "_" + str(id_rsu)
            RSU_NF._ids[name_rsu] += 1

        super(RSU_NF, self).__init__(name=name)
        self.n = n
        self.filters = filters
        self.filters_out = filters_out

        self.conv_in = REBNCONV(self.filters_out, dilation_rate=1)
        self.conv_encoder = [REBNCONV(self.filters, dilation_rate=2 ** i) for i in range(self.n - 1)]

        self.conv_n = REBNCONV(self.filters, dilation_rate=2 ** (self.n - 1))

        self.conv_decoder = [REBNCONV(self.filters, dilation_rate=2 ** i) for i in range(1, self.n - 1)]
        self.conv_out = REBNCONV(self.filters_out, dilation_rate=1)

    def call(self, inputs, training=None):
        xin = self.conv_in(inputs, training=training)

        # encoder
        x_is = []
        x = xin
        for i in range(self.n - 1):
            x = self.conv_encoder[i](x, training=training)
            x_is.append(x)

        # mid
        x = self.conv_n(x_is[-1], training=training)

        # decoder
        for i in range(self.n - 3, -1, -1):
            x_i = x_is[i + 1]
            x = kl.Concatenate()([x, x_i])
            x = self.conv_decoder[i](x, training=training)

        x = kl.Concatenate()([x, x_is[0]])
        x = self.conv_out(x, training=training)
        return xin + x

    def get_config(self):
        config = {
            'n': self.n,
            'filters': self.filters,
            'filters_out': self.filters_out,
        }
        base_config = super(RSU_NF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def U2Net(input_shape):
    inputs = kl.Input(input_shape)

    # Encoder
    x1 = RSU_N(7, 32, 64)(inputs)  # RSU-7
    x = kl.MaxPool2D(2, 2)(x1)

    x2 = RSU_N(6, 32, 128)(x)  # RSU-6
    x = kl.MaxPool2D(2, 2)(x2)

    x3 = RSU_N(5, 64, 256)(x)  # RSU-5
    x = kl.MaxPool2D(2, 2)(x3)

    x4 = RSU_N(4, 128, 512)(x)  # RSU-5
    x = kl.MaxPool2D(2, 2)(x4)

    x5 = RSU_NF(4, 256, 512)(x)  # RSU-4F
    x = kl.MaxPool2D(2, 2)(x5)

    x6 = RSU_NF(4, 256, 512)(x)  # RSU-4F
    x = kl.UpSampling2D(size=[2, 2], interpolation="bilinear")(x6)

    # Decoder
    x = kl.Concatenate()([x, x5])
    x5d = RSU_NF(4, 256, 512)(x)  # RSU-4F
    x = kl.UpSampling2D(size=[2, 2], interpolation="bilinear")(x5d)

    x = kl.Concatenate()([x, x4])
    x4d = RSU_N(4, 128, 256)(x)  # RSU-4
    x = kl.UpSampling2D(size=[2, 2], interpolation="bilinear")(x4d)

    x = kl.Concatenate()([x, x3])
    x3d = RSU_N(5, 64, 128)(x)  # RSU-5
    x = kl.UpSampling2D(size=[2, 2], interpolation="bilinear")(x3d)

    x = kl.Concatenate()([x, x2])
    x2d = RSU_N(6, 32, 64)(x)  # RSU-6
    x = kl.UpSampling2D(size=[2, 2], interpolation="bilinear")(x2d)

    x = kl.Concatenate()([x, x1])
    x1d = RSU_N(7, 16, 64)(x)  # RSU-7

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
