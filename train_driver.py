import datetime
import glob
import os
from functools import partial

import tensorflow as tf
import tensorflow_addons as tfa

from data_processing import read_sample, preprocess_sample, augment
from u2net import U2Net

USE_MIXED_PRECISION = False
USE_MULTI_GPU = True
RESIZE_SHAPE = (320, 320)
CROP_SIZE = (288, 288)
BATCH_SIZE_PER_DEVICE = 36

TRAIN_IMG_PATH = "/home/crr/datasets/duts/DUTS-TR/DUTS-TR-Image"
TRAIN_MASK_PATH = "/home/crr/datasets/duts/DUTS-TR/DUTS-TR-Mask"
TEST_IMG_PATH = "/home/crr/datasets/duts/DUTS-TE/DUTS-TE-Image"
TEST_MASK_PATH = "/home/crr/datasets/duts/DUTS-TE/DUTS-TE-Mask"

train_img_paths = glob.glob(os.path.join(TRAIN_IMG_PATH, "*.*"))
train_mask_paths = [os.path.join(TRAIN_MASK_PATH, os.path.splitext(os.path.basename(imgp))[0] + ".png") for imgp in
                    train_img_paths]
test_img_paths = glob.glob(os.path.join(TEST_IMG_PATH, "*.*"))
test_mask_paths = [os.path.join(TEST_MASK_PATH, os.path.splitext(os.path.basename(imgp))[0] + ".png") for imgp in
                   test_img_paths]

# %% LOAD DATA
if USE_MIXED_PRECISION:
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print("Computation dtype:", policy.compute_dtype)
    print("Variable dtype:", policy.variable_dtype)

if USE_MULTI_GPU:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

global_batch_size = BATCH_SIZE_PER_DEVICE * strategy.num_replicas_in_sync

train_td = tf.data.Dataset.from_tensor_slices((train_img_paths, train_mask_paths))
train_td = train_td.map(read_sample, num_parallel_calls=-1)
train_td = train_td.map(partial(preprocess_sample, size=RESIZE_SHAPE), num_parallel_calls=-1)
train_td = train_td.batch(global_batch_size)
train_td = train_td.map(partial(augment, crop_size=CROP_SIZE, seed=42), num_parallel_calls=-1)
train_td = train_td.repeat(-1)
train_td = train_td.prefetch(-1)

test_td = tf.data.Dataset.from_tensor_slices((test_img_paths, test_mask_paths))
test_td = test_td.map(read_sample, num_parallel_calls=-1)
test_td = test_td.map(partial(preprocess_sample, size=RESIZE_SHAPE), num_parallel_calls=-1)
test_td = test_td.batch(global_batch_size)
test_td = test_td.prefetch(-1)

# %% MAKE MODEL
with strategy.scope():
    u2net = U2Net(input_shape=[None, None, 3])
    u2net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-8),
                  loss=[tf.keras.losses.BinaryCrossentropy()] * 7,
                  metrics={
                      "s0": [
                          tf.keras.metrics.MeanIoU(2, name="iou"),
                          tfa.metrics.FBetaScore(2, beta=0.3, average="micro", name="Fbeta"),
                          tf.keras.metrics.MeanAbsoluteError(name="mae")
                      ]
                  }
                  )
u2net.summary()

# %% TRAIN
total_steps = 600000 // (global_batch_size // 12)
steps_per_epoch = 100
epochs = total_steps // steps_per_epoch

output_dir = os.path.join("outputs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(output_dir)
hist = u2net.fit(train_td,
                 epochs=epochs,
                 steps_per_epoch=steps_per_epoch,
                 callbacks=[
                     tf.keras.callbacks.CSVLogger(
                         filename=os.path.join(output_dir, "log.csv")
                     ),
                     tf.keras.callbacks.ModelCheckpoint(
                         filepath=os.path.join(output_dir, "weights.h5"),
                         monitor="s0_Fbeta",
                         mode="max",
                         save_weights_only=True
                     ),
                     tf.keras.callbacks.TensorBoard(log_dir=output_dir, profile_batch=0)
                     # ssh -L 6006:127.0.0.1:6006 <user>@<ip>
                 ]
                 )

# float32, 36bz - 105 s
# float16, 36bz - *

# %% EVALUATE
u2net.evaluate(test_td)
