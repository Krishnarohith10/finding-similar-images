"""
This is a python file where I trained the model from model created in 'creating_a_model' 
file. Tensorflow version=1.14.0, cause I'm getting issues in version 2.1.0, so please
excuse me. I enabled eager execution so. Finding Similar images from given image means finding
similarity function, which can be cosine or the popular triplet fucntion. In this I used triplet
loss for monitoring the loss os the model. Since our goal is to increase the similarity between 
anchor and positive, decreasing the similarity between anchor and negative. It;s pretty simple, 
I took dataset, created 'images' dictionary with classes as keys and images as values. Now 
creating anchor, positive and negative data. Creating an instance for our created model. There
are three created models, which takes three inputs, anchors, positives, negatives and finds
128-d vector, this network is known as siamese network. This is mainly for face recognition
purpose also called as one-shot learning. 
"""
import os # library for moving directory to directory
import cv2 # opencv library for operations on images
import pickle # used to load .pickle file and dump .pickle files
import numpy as np # operations on array (high dimensional also)
import tensorflow.compat.v1 as tf # tensorflow framework for deep learning algorithms
import tensorflow.python.keras.api.keras as keras # keras framework for deep learning algorithms
from creating_a_model import faceRecogModel # calling faceRecogModel function created in created-a_model.py file

tf.enable_eager_execution()

with open('.\\path_dict.pickle', 'rb') as f:
    paths = pickle.load(f)

faces = list(paths.keys())

# the dataset used here is modified from given dataset. creating folder 'dataset_train' with 
# classes as subfolders and all images per class per subfolder this is created from 
# dataset folder 'dataset_train'.
images = {}
for key in paths.keys():
    li=[]
    for image in os.listdir(paths[key]):
        img = cv2.imread(os.path.join(paths[key], image))
        li.append(img)
    images[key]=np.array(li)/255.0
    print('Data Acquired Completed! Key:', key, 'Number Of Images:', len(li))
# deleting unnecessary variables which takes RAM and we definitely need more RAM
del li, img

batch_size = 64
input_shape = (96, 96, 3)

# creating batch generator for training our model. anchors, positives, negatives with shape
# batch_size, image shape. Here we require output label (y) so we're sending zero vector of 
# same batch_size, 128-d shape.
def next_batch(batch_size):
    y = np.zeros((batch_size, 128))
    anchors = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    positives = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    negatives = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    
    while True:
        for i in range(batch_size):
            positive_face = faces[np.random.randint(len(faces))]
            negative_face = faces[np.random.randint(len(faces))]
            while positive_face == negative_face:
                negative_face = faces[np.random.randint(len(faces))]
            positives[i] = images[positive_face][np.random.randint(len(images[positive_face]))]
            anchors[i] = images[positive_face][np.random.randint(len(images[positive_face]))]
            while positives[i].all() == anchors[i].all():
                anchors[i] = images[positive_face][np.random.randint(len(images[positive_face]))]
            negatives[i] = images[negative_face][np.random.randint(len(images[negative_face]))]
        
        yield ([anchors, positives, negatives], [y, y, y])

# function for triplet loss, here we don't require y_true, so neglect it. and the mathemetical
# equation used is for triplet loss itself.
def triplet_loss(y_true, y_pred, alpha=0.5):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss

#creating an instance for our created model which we will use to train using given 
# data with input_shape=(96, 96, 3)
model = faceRecogModel(input_shape=(96, 96, 3))

# As we know triplet model takes three inputs anchors, positives, negatives and train them with
# our data and predict the output vec. So we should provide with three different inputs.
input_shape=(96, 96, 3)
inp_A = keras.Input(shape=input_shape)
inp_P = keras.Input(shape=input_shape)
inp_N = keras.Input(shape=input_shape)

# same model model is trained for three inputs which gives three different encodings vector.
enc_A = model(inp_A)
enc_P = model(inp_P)
enc_N = model(inp_N)

# early stopping, if model loss doesn't change for 5 epochs. And Learning rate scheduler
# to change learning rate 
early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.00005)

#creating a triplet model which takes 3 inputs and gives three outputs and compiling
# with loss given as triplet loss, cause our aim to decrease triplet loss not anyother loss.
tripletModel = keras.models.Model(inputs=[inp_A, inp_P, inp_N], outputs=[enc_A, enc_P, enc_N])
rmspropoptimizer = keras.optimizers.RMSprop(learning_rate=0.1, momentum=0.9, epsilon=1.0)

tripletModel.compile(optimizer = rmspropoptimizer, loss = triplet_loss)

# creating a batch generator for training our model.
generator = next_batch(batch_size)

#finally training our model on generator created.
history = tripletModel.fit_generator(generator, epochs=10, steps_per_epoch=1, callbacks=[early_stopping])

# we need to save the the trained model which is model not triplet model. Cause triplet model is used for 
# tunning the paraters and learning through it. And when triplet model is learnt, then model
# will have the parameters same as triplet model cause triplet mode uses model with just three different 
# inputs. And our predicted image will be only one not many (anchors, positives, negatives).
print('Saving your model as jpg file........', end=' ')
keras.utils.plot_model(model, to_file='faceRecogModel-short.jpg')
print('Saved!')
print('Model saving .........', end=' ')
name = 'faceRecogModel-short.h5'
model.save(name)
print('Saved at path:', os.path.join(os.getcwd(), name))
