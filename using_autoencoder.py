# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:55:42 2020

@author: krish
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.api.keras.callbacks import EarlyStopping
from tensorflow.python.keras.api.keras.models import Sequential, load_model
from tensorflow.python.keras.api.keras.layers import Input, Conv2D, Conv2DTranspose

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

path='D:\\finding_similar_images\\dataset'
images = []
for image in tqdm(os.listdir(path)):
    img = cv2.imread(os.path.join(path, image))
    img = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
    images.append(img)

images = np.array(images).astype('float32')/255.0

indexes = np.random.permutation(len(images))
images = images[indexes]

x_train, x_test = train_test_split(images, test_size=0.2)

del images

autoencoder = Sequential()

autoencoder.add(Input(shape=(64,64,3)))
autoencoder.add(Conv2D(16, (3,3), (2,2), padding='same', activation='relu'))
autoencoder.add(Conv2D(16, (3,3), (2,2), padding='same', activation='relu'))
autoencoder.add(Conv2D(8, (3,3), (2,2), padding='same', activation='relu'))
autoencoder.add(Conv2D(8, (3,3), (2,2), padding='same', activation='relu'))
autoencoder.add(Conv2DTranspose(8, (3,3), (2,2), padding='same', activation='relu'))
autoencoder.add(Conv2DTranspose(8, (3,3), (2,2), padding='same', activation='relu'))
autoencoder.add(Conv2DTranspose(16, (3,3), (2,2), padding='same', activation='relu'))
autoencoder.add(Conv2DTranspose(16, (3,3), (2,2), padding='same', activation='relu'))
autoencoder.add(Conv2D(3, (3,3), (1,1), padding='same', activation='sigmoid'))

autoencoder.compile(optimizer='adam', loss='mae')

early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=5, 
                               mode='min', 
                               restore_best_weights=True)

epochs = 100
batch_size = 64
history = autoencoder.fit(x_train, x_train, batch_size=batch_size, 
                          callbacks=[early_stopping], epochs=epochs, 
                          verbose=1, validation_split=0.1)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Loss')
plt.legend()
plt.show()

autoencoder.save('autoncoder.h5')

del autoencoder

autoencoder = load_model('autoncoder.h5')

x_train_embeddings = autoencoder.predict(x_train)

x_train_embeddings_flatten = x_train_embeddings.reshape((len(x_train_embeddings), -1))

nn = NearestNeighbors(n_neighbors=5)
nn.fit(x_train_embeddings_flatten)

idx = np.random.randint(len(x_test))
img_query = x_test[idx]

indices = nn.kneighbors([x_train_embeddings_flatten[idx]], return_distance=False)

img_retrieval = [x_train[idx] for idx in indices.flatten()]

plt.figure()
plt.imshow(cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB))
plt.show()

for i, img in enumerate(img_retrieval):
    plt.subplot(5,1,i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
