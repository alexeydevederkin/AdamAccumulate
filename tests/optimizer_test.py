import keras.backend as K
import numpy as np
import tensorflow as tf
import random as rn

# Reproducibility
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

# run it with env variables: $ CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python3 optimizer_test.py

np.random.seed(42)
rn.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


from keras import models, layers

model = models.Sequential()

model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), _ = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
train_labels = to_categorical(train_labels)


model_2 = models.clone_model(model)
model_2.set_weights(model.get_weights())

model_3 = models.clone_model(model)
model_3.set_weights(model.get_weights())


optimizer = Adam(lr=0.0001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

print('\nTraining with Adam, 1st run:')
model.fit(train_images, train_labels, epochs=5, batch_size=32, shuffle=False)


optimizer_2 = Adam(lr=0.0001)
model_2.compile(
    optimizer=optimizer_2,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

print('\nTraining with Adam, 2nd run:')
model_2.fit(train_images, train_labels, epochs=5, batch_size=32, shuffle=False)


optimizer_3 = AdamAccumulate(lr=0.0001, accum_iters=8)
model_3.compile(
    optimizer=optimizer_3,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

print('\nTraining with AdamAccumulate:')
model_3.fit(train_images, train_labels, epochs=5, batch_size=4, shuffle=False)