import glob
import os
from functools import partial

import tensorflow as tf

from data.data_processing import read_sample, preprocess_sample
from data.tf_record import tf_serialize_tensor_example

RESIZE_SHAPE = (320, 320)

# IMG_PATH = "/home/crr/datasets/duts/DUTS-TR/DUTS-TR-Image"
# MASK_PATH = "/home/crr/datasets/duts/DUTS-TR/DUTS-TR-Mask"
IMG_PATH = "/home/crr/datasets/duts/DUTS-TE/DUTS-TE-Image"
MASK_PATH = "/home/crr/datasets/duts/DUTS-TE/DUTS-TE-Mask"

img_paths = glob.glob(os.path.join(IMG_PATH, "*.*"))
mask_paths = [os.path.join(MASK_PATH, os.path.splitext(os.path.basename(imgp))[0] + ".png") for imgp in
              img_paths]

# %%
td = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
td = td.map(read_sample, num_parallel_calls=-1)
td = td.map(partial(preprocess_sample, size=RESIZE_SHAPE, normalize=True), num_parallel_calls=-1)
td = td.map(tf_serialize_tensor_example)
td = td.prefetch(-1)

# %%
# output_record = "/home/crr/datasets/duts/DUTS-TR/train.tfrecord"
output_record = "/home/crr/datasets/duts/DUTS-TE/test.tfrecord"

writer = tf.data.experimental.TFRecordWriter(output_record)
writer.write(td)
