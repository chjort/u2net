import datetime
import os
from functools import partial

import tensorflow as tf
import tensorflow_addons as tfa

from data.data_processing import augment
from data.tf_record import batch_deserialize_tensor_example_float32
from u2net import U2Net

USE_MIXED_PRECISION = True
USE_MULTI_GPU = True
RESIZE_SHAPE = (320, 320)
CROP_SIZE = (288, 288)
BATCH_SIZE_PER_DEVICE = 36

TRAIN_PATH = "/home/crr/datasets/duts/DUTS-TR/train.tfrecord"
TEST_PATH = "/home/crr/datasets/duts/DUTS-TE/test.tfrecord"

# %% LOAD DATA
if USE_MIXED_PRECISION:
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    tf.keras.mixed_precision.experimental.set_policy(policy)
    BATCH_SIZE_PER_DEVICE = BATCH_SIZE_PER_DEVICE * 2
    print("Computation dtype:", policy.compute_dtype)
    print("Variable dtype:", policy.variable_dtype)

if USE_MULTI_GPU:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

global_batch_size = BATCH_SIZE_PER_DEVICE * strategy.num_replicas_in_sync

train_td = tf.data.TFRecordDataset(TRAIN_PATH, num_parallel_reads=-1)
train_td = train_td.batch(global_batch_size)
train_td = train_td.map(batch_deserialize_tensor_example_float32)
train_td = train_td.map(partial(augment, crop_size=CROP_SIZE, seed=42), num_parallel_calls=-1)
train_td = train_td.repeat(-1)
train_td = train_td.prefetch(-1)

test_td = tf.data.TFRecordDataset(TEST_PATH, num_parallel_reads=-1)
test_td = test_td.batch(global_batch_size)
test_td = test_td.map(batch_deserialize_tensor_example_float32)
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
log_dir = os.path.join(output_dir, "logs")
weights_dir = os.path.join(output_dir, "weights")
os.makedirs(log_dir)
os.makedirs(weights_dir)

u2net.save_weights(os.path.join(weights_dir, "initial_weights.h5"))
hist = u2net.fit(train_td,
                 epochs=epochs,
                 steps_per_epoch=steps_per_epoch,
                 callbacks=[
                     tf.keras.callbacks.CSVLogger(
                         filename=os.path.join(log_dir, "log.csv")
                     ),
                     tf.keras.callbacks.ModelCheckpoint(
                         filepath=os.path.join(weights_dir, "best_weights.h5"),
                         monitor="s0_Fbeta",
                         mode="max",
                         save_weights_only=True,
                         save_best_only=True
                     ),
                     tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)
                     # ssh -L 6006:127.0.0.1:6006 <user>@<ip>
                 ]
                 )

# %% EVALUATE
u2net.save(os.path.join(weights_dir, "model.h5"))
u2net.save_weights(os.path.join(weights_dir, "latest_weights.h5"))

u2net.evaluate(test_td)
