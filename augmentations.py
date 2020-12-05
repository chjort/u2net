import tensorflow as tf

tf.keras.backend.int_shape
def resize(image, min_side=800, max_side=1333):
    """
    image with rank 3 [h, w, c]
    """

    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)

    cur_min_side = tf.minimum(w, h)
    min_side = tf.cast(min_side, tf.float32)

    if max_side is not None:
        cur_max_side = tf.maximum(w, h)
        max_side = tf.cast(max_side, tf.float32)
        scale = tf.minimum(max_side / cur_max_side,
                           min_side / cur_min_side)
    else:
        scale = min_side / cur_min_side

    nh = tf.cast(scale * h, tf.int32)
    nw = tf.cast(scale * w, tf.int32)

    image = tf.image.resize(image, (nh, nw))
    return image
