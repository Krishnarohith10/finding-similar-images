"""
This is MAIN python file which we need to run. In this we see the predictions of our model on new data and find similar images
with cosine distance from training data available.
"""
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split # scikit-learn library for machine learning algorithms
from sklearn.neighbors import NearestNeighbors # K Nearest Neighbors machine learning classification algorithm
import tensorflow.compat.v1 as tf
import tensorflow.python.keras.api.keras as keras
from data_plotting import plot_query_retrieval #ploting images of query and similar images

print('Tensorflow version:', tf.__version__, 'Keras version:', keras.__version__)
tf.enable_eager_execution()

# uses dataset provided by you but named 'dataset_predict' and loading it to 'data'.
data = []
path = '.\\dataset_predict'
print('Loading data from', path, '.........', end=' ')
for image in os.listdir(path):
    img = cv2.imread(os.path.join(path, image))
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
    data.append(img)
print('Data loaded!')
data = np.array(data)

#loading trained model 'faceRecogModel.h5'. Here we get an warning for not compiling this model
# remember we compiled and trained triplet model not FRModel (we trained FRModel indirectly).
print('Loading model.........', end=' ')
faceRecogModel = keras.models.load_model('faceRecogModel.h5')
print('Model', faceRecogModel.name, 'loaded!')

#spliting the data to x_train, x_test with test_size 2% and rest train_size. cause we don't want to waste 
# alot of data for just test. Instead you can also use whole data for predicting and then 
# use an new image or images for testing and plotting.
x_train, x_test = train_test_split(data, test_size=0.02, random_state=42)

# feature scaling, very important for Neural Network. Used for bringing all feature to one scale
# instead of each having different scales.
x_train_scaled = x_train/255
x_test_scaled = x_test/255
# finally predicting using faceRecogModel.predict()
print('Predicting on dataset...........', end=' ')
x_train_embeddings = faceRecogModel.predict(x_train_scaled)
x_test_embeddings = faceRecogModel.predict(x_test_scaled)
print('Prediction Completed!')

# creating an object for nearestNeighbors with finding n_neighbors(9) similar images to given image with 
# metric = 'cosine' function.
print('Creating NearestNeighbors Classifier.........', end=' ')
neigh = NearestNeighbors(n_neighbors=9, metric='cosine')
print('Created!')

# fitting this neigh object to our x_train data to learn from this data.
print('Fitting NearestNeighbors on predictions........', end=' ')
neigh.fit(x_train_embeddings)
print('Completed')
#now it may get messy but I created a list(well two), which stores query images (test images)
# the other stores the similar images (5 images) per each test image (query image). (test==query)
# for our demonstartion we are only using first num=10 of x_test images and plotting.

num = np.random.randint(len(x_test_embeddings))
print('Finding similar images using NearestNeighbors for random image (num) in test data')
indices = neigh.kneighbors([x_test_embeddings[num]], return_distance=False)
img_query = x_test[num]
img_retrieval = [x_train[idx] for idx in indices]
print('Completed!')

# finally calls the function created in 'data_plotting.py' file. And plots the images.
plot_query_retrieval(img_query, img_retrieval)

