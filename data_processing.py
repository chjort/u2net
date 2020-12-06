import matplotlib.pyplot as plt
import tensorflow as tf


def read_image(img_path, channels=3):
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_bytes, channels=channels)
    img.set_shape([None, None, channels])
    return img


def imshow(img):
    plt.imshow(img)
    plt.show()


def imshow_sample(x, y):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(x)
    ax2.imshow(y)
    plt.show()


def read_sample(x, y):
    x = read_image(x, channels=3)
    y = read_image(y, channels=1)
    return x, y


def preprocess_sample(x, y, size=(320, 320)):
    x_channels = tf.shape(x)[-1]
    xy = tf.concat([x, y], axis=-1)

    xy = preprocess_img(xy, size=size)

    x = xy[..., :x_channels]
    y = xy[..., x_channels:]

    return x, y


def preprocess_img(x, size=(320, 320)):
    x = tf.cast(x, tf.float32)
    x = x / 255.
    x = tf.image.resize(x, size=size)
    return x


def augment(x, y, crop_size=(288, 288), seed=None):
    x_channels = tf.shape(x)[-1]
    xy = tf.concat([x, y], axis=-1)

    xy = tf.image.random_flip_left_right(xy, seed=seed)

    shape = tf.shape(xy)
    xy = tf.image.random_crop(xy, size=(shape[0], *crop_size, shape[3]), seed=seed)

    x = xy[..., :x_channels]
    y = xy[..., x_channels:]

    return x, y
