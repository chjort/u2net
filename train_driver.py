import glob
import os
from functools import partial

import tensorflow as tf

from data_processing import imshow, read_sample, preprocess_sample, augment

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

# %%
train_td = tf.data.Dataset.from_tensor_slices((train_img_paths, train_mask_paths))
train_td = train_td.map(read_sample)
train_td = train_td.map(partial(preprocess_sample, size=[320, 320]))
train_td = train_td.batch(12)
train_td = train_td.map(partial(augment, seed=42))

test_td = tf.data.Dataset.from_tensor_slices((test_img_paths, test_mask_paths))
test_td = test_td.map(read_sample)
test_td = test_td.map(partial(preprocess_sample, size=[320, 320]))
test_td = test_td.batch(12)

# %%
it = iter(train_td)
x, y = next(it)

print(x.shape, y.shape)

idx = 3
imshow(x[idx])
imshow(y[idx])
