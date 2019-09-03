from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

print(tf.__version__)

import pathlib
#data_root = pathlib.Path('/content/drive/My Drive/Images2')
data_root = pathlib.Path(r'C:\Users\EchoY\OneDrive\Desktop\ProjectData\train\images2')
test_root = pathlib.Path(r'C:\Users\EchoY\OneDrive\Desktop\ProjectData\train\images2_test')
print(data_root)
print(test_root)

import random
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
print(image_count)

all_test_paths = list(test_root.glob('*/*'))
all_test_paths = [str(path) for path in all_test_paths]
random.shuffle(all_test_paths)

test_count = len(all_test_paths)
print(test_count)

image_size = 192 # All images will be resized to 192x192
batch_size = 32

# Rescale all images by 1./255 and apply image augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    directory=data_root,
    target_size=(image_size, image_size),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    directory=data_root,
    target_size=(image_size, image_size),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset='validation'
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory=test_root,
    target_size=(image_size, image_size),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()
    
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(image_size, image_size, 3)))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss = keras.losses.categorical_crossentropy,
                     optimizer='adam',
                     metrics=['accuracy'])
                     
                     
batch_stats_callback = CollectBatchStats()

checkpoint_path = r"C:\Users\EchoY\OneDrive\Desktop\ProjectData\train\conv_model\cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit_generator(train_generator,
                    steps_per_epoch = train_generator.samples // batch_size,
                    validation_data = validation_generator, 
                    validation_steps = validation_generator.samples // batch_size,
                    epochs = 10,
                    callbacks = [cp_callback, batch_stats_callback])
model.save(r'C:\Users\EchoY\OneDrive\Desktop\ProjectData\train\conv_model\my_model.h5')

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

model.evaluate_generator(generator=validation_generator,
steps=STEP_SIZE_VALID)
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv(r"C:\Users\EchoY\OneDrive\Desktop\ProjectData\train\results.csv",index=False)
