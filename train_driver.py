import datetime
import glob
import os
from functools import partial

import tensorflow as tf
import tensorflow_addons as tfa

from data_processing import read_sample, preprocess_sample, augment
from u2net import U2Net

USE_MIXED_PRECISION = False
RESIZE_SHAPE = (320, 320)
CROP_SIZE = (288, 288)

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
train_td = tf.data.Dataset.from_tensor_slices((train_img_paths, train_mask_paths))
train_td = train_td.map(read_sample, num_parallel_calls=-1)
train_td = train_td.map(partial(preprocess_sample, size=RESIZE_SHAPE), num_parallel_calls=-1)
train_td = train_td.batch(12)
train_td = train_td.map(partial(augment, crop_size=CROP_SIZE, seed=42), num_parallel_calls=-1)
train_td = train_td.repeat(-1)
train_td = train_td.prefetch(-1)

test_td = tf.data.Dataset.from_tensor_slices((test_img_paths, test_mask_paths))
test_td = test_td.map(read_sample, num_parallel_calls=-1)
test_td = test_td.map(partial(preprocess_sample, size=RESIZE_SHAPE), num_parallel_calls=-1)
test_td = test_td.batch(12)
test_td = test_td.prefetch(-1)

# %% MAKE MODEL
if USE_MIXED_PRECISION:
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print("Computation dtype:", policy.compute_dtype)
    print("Variable dtype:", policy.variable_dtype)

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
TOTAL_STEPS = 600000
STEPS_PER_EPOCH = 1000
EPOCHS = TOTAL_STEPS // STEPS_PER_EPOCH

output_dir = os.path.join("outputs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(output_dir)
hist = u2net.fit(train_td,
                 epochs=EPOCHS,
                 steps_per_epoch=STEPS_PER_EPOCH,
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

# %% EVALUATE
u2net.evaluate(test_td)
