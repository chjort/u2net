import tensorflow as tf

_SERIALIZE_IMAGE_TENSOR_DESCRIPTION = {
    "x": tf.io.FixedLenFeature([], tf.string),
    "y": tf.io.FixedLenFeature([], tf.string),
}


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a byte value."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_tensor_example(x, y):
    feature = {
        "x": _bytes_feature(tf.io.serialize_tensor(x)),
        "y": _bytes_feature(tf.io.serialize_tensor(y)),
    }

    sample_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return sample_proto.SerializeToString()


def tf_serialize_tensor_example(x, y):
    example = tf.py_function(
        serialize_tensor_example,
        (x, y),
        tf.string
    )
    return tf.reshape(example, [])


def batch_deserialize_tensor_example(x, dtype=tf.float32):
    tensor_example = tf.io.parse_example(x, _SERIALIZE_IMAGE_TENSOR_DESCRIPTION)

    @tf.function
    def _parse_tensor_dtype(x):
        return tf.io.parse_tensor(x, out_type=dtype)

    x = tf.map_fn(_parse_tensor_dtype, tensor_example["x"],
                  fn_output_signature=tf.TensorSpec(shape=[None, None, 3], dtype=dtype),
                  parallel_iterations=8,
                  infer_shape=False)
    y = tf.map_fn(_parse_tensor_dtype, tensor_example["y"],
                  fn_output_signature=tf.TensorSpec(shape=[None, None, 1], dtype=dtype),
                  parallel_iterations=8,
                  infer_shape=False)

    return x, y


def batch_deserialize_tensor_example_uint8(x):
    return batch_deserialize_tensor_example(x, tf.uint8)


def batch_deserialize_tensor_example_int32(x):
    return batch_deserialize_tensor_example(x, tf.int32)


def batch_deserialize_tensor_example_int64(x):
    return batch_deserialize_tensor_example(x, tf.int64)


def batch_deserialize_tensor_example_float32(x):
    return batch_deserialize_tensor_example(x, tf.float32)


def batch_deserialize_tensor_example_float64(x):
    return batch_deserialize_tensor_example(x, tf.float64)


def deserialize_tensor_example(x, dtype=tf.float32):
    tensor_example = tf.io.parse_single_example(x, _SERIALIZE_IMAGE_TENSOR_DESCRIPTION)

    x = tf.io.parse_tensor(tensor_example["x"], out_type=dtype)
    y = tf.io.parse_tensor(tensor_example["y"], out_type=dtype)

    x.set_shape([None, None, 3])
    y.set_shape([None, None, 1])
    return x, y


def deserialize_tensor_example_uint8(x):
    return deserialize_tensor_example(x, tf.uint8)


def deserialize_tensor_example_int32(x):
    return deserialize_tensor_example(x, tf.int32)


def deserialize_tensor_example_int64(x):
    return deserialize_tensor_example(x, tf.int64)


def deserialize_tensor_example_float32(x):
    return deserialize_tensor_example(x, tf.float32)


def deserialize_tensor_example_float64(x):
    return deserialize_tensor_example(x, tf.float64)
