import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
import PIL
img_dims=(300, 300)
batch_size=126
Data_dir='../RPC/'

train_Data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
      validation_split=0.5)

train_generator = train_Data_gen.flow_from_directory(
	Data_dir+'rps',
	target_size=(150,150),
	class_mode='categorical',
  batch_size=batch_size
)

val_Data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

validation_generator = val_Data_gen.flow_from_directory(
	Data_dir+'rps-test-set',
	target_size=(150,150),
	class_mode='categorical',
  batch_size=batch_size
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)
'''
Epoch 1/25
20/20 [==============================] - 27s 975ms/step - loss: 1.4726 - accuracy: 0.3431 - val_loss: 1.0960 - val_accuracy: 0.3333
Epoch 2/25
20/20 [==============================] - 20s 993ms/step - loss: 1.1106 - accuracy: 0.3708 - val_loss: 0.9326 - val_accuracy: 0.6747
Epoch 3/25
20/20 [==============================] - 20s 1s/step - loss: 1.1381 - accuracy: 0.4770 - val_loss: 0.7427 - val_accuracy: 0.6452
Epoch 4/25
20/20 [==============================] - 20s 999ms/step - loss: 0.9135 - accuracy: 0.5597 - val_loss: 0.6028 - val_accuracy: 0.9328
Epoch 5/25
20/20 [==============================] - 20s 973ms/step - loss: 0.8508 - accuracy: 0.6647 - val_loss: 0.3537 - val_accuracy: 0.9892
Epoch 6/25
20/20 [==============================] - 21s 1s/step - loss: 0.7915 - accuracy: 0.6793 - val_loss: 1.0450 - val_accuracy: 0.4946
Epoch 7/25
20/20 [==============================] - 20s 983ms/step - loss: 0.7646 - accuracy: 0.6721 - val_loss: 0.9614 - val_accuracy: 0.4543
Epoch 8/25
20/20 [==============================] - 21s 1s/step - loss: 0.6460 - accuracy: 0.7017 - val_loss: 0.1584 - val_accuracy: 0.9435
Epoch 9/25
20/20 [==============================] - 19s 952ms/step - loss: 0.4385 - accuracy: 0.8175 - val_loss: 0.1560 - val_accuracy: 0.9462
Epoch 10/25
20/20 [==============================] - 18s 913ms/step - loss: 0.3713 - accuracy: 0.8529 - val_loss: 0.1584 - val_accuracy: 0.9731
Epoch 11/25
20/20 [==============================] - 19s 919ms/step - loss: 0.2811 - accuracy: 0.9072 - val_loss: 0.3320 - val_accuracy: 0.8065
Epoch 12/25
20/20 [==============================] - 19s 923ms/step - loss: 0.3062 - accuracy: 0.8865 - val_loss: 0.0445 - val_accuracy: 0.9812
Epoch 13/25
20/20 [==============================] - 17s 840ms/step - loss: 0.2182 - accuracy: 0.9144 - val_loss: 0.3387 - val_accuracy: 0.8333
Epoch 14/25
20/20 [==============================] - 18s 871ms/step - loss: 0.3211 - accuracy: 0.8718 - val_loss: 0.1948 - val_accuracy: 0.9167
Epoch 15/25
20/20 [==============================] - 19s 950ms/step - loss: 0.1633 - accuracy: 0.9423 - val_loss: 0.0682 - val_accuracy: 0.9785
Epoch 16/25
20/20 [==============================] - 19s 931ms/step - loss: 0.2181 - accuracy: 0.9205 - val_loss: 0.1246 - val_accuracy: 0.9462
Epoch 17/25
20/20 [==============================] - 19s 930ms/step - loss: 0.1177 - accuracy: 0.9628 - val_loss: 0.0322 - val_accuracy: 1.0000
Epoch 18/25
20/20 [==============================] - 18s 907ms/step - loss: 0.1480 - accuracy: 0.9530 - val_loss: 0.0546 - val_accuracy: 0.9677
Epoch 19/25
20/20 [==============================] - 18s 902ms/step - loss: 0.0937 - accuracy: 0.9662 - val_loss: 0.1434 - val_accuracy: 0.9113
Epoch 20/25
20/20 [==============================] - 18s 893ms/step - loss: 0.1531 - accuracy: 0.9484 - val_loss: 0.0551 - val_accuracy: 0.9839
Epoch 21/25
20/20 [==============================] - 19s 938ms/step - loss: 0.0952 - accuracy: 0.9682 - val_loss: 0.0505 - val_accuracy: 0.9785
Epoch 22/25
20/20 [==============================] - 19s 923ms/step - loss: 0.1192 - accuracy: 0.9604 - val_loss: 0.3483 - val_accuracy: 0.8548
Epoch 23/25
20/20 [==============================] - 19s 923ms/step - loss: 0.2081 - accuracy: 0.9191 - val_loss: 0.0468 - val_accuracy: 0.9785
Epoch 24/25
20/20 [==============================] - 18s 895ms/step - loss: 0.1886 - accuracy: 0.9382 - val_loss: 0.2530 - val_accuracy: 0.9005
Epoch 25/25
20/20 [==============================] - 18s 904ms/step - loss: 0.0887 - accuracy: 0.9713 - val_loss: 0.6816 - val_accuracy: 0.7769
'''
