# finding-similar-images

The model used for training is faceRecogModel, which is familar model. It is used to find the 128-dimentional vector. This model is used for face recognition purpose. 

I intend to use this model because in this project we are finding similar images for given query image, which means we are claculating triplet loss. Triplet loss is the loss between an anchor, positive and negative image. It is used to increase the similarity between anchor and positive image and decrease the similarity between anchor and negative image.

You need to prepare dataset first. Download the dataset here: https://drive.google.com/file/d/1VT-8w1rTT2GCE5IE5zFJPMzv7bqca-Ri/view?usp=sharing 

create a two dataset folders, 'dataset_train'and 'dataset_predict' respectively. In dataset_train folder, you find subfolder with classes as names and images per each class in this folder. For Example: 

* dataset_train
  * cheetah
    * 1.jpg
    * 2.jpg
  * tiger
    * 1.jpg
    * 2.jpg
  *
    *
    *
* dataset_predict
  * 1.jpg
  * 2.jpg
  *
  *


**faceRecogModel.h5** is trained model for dataset_train. All you have to do is run the file **predictions_and_finding_similar_images.py** file.

**creating_a_model.py** - is used to create a Model, in this we use faceRecogModel.

**training_created_model.py** - Since we're working on triplet loss, we need to get triplets dataset. So we will created three different data variables, anchors, positives, negatives. Ecah of (batch_size, shape of input image) [For Ex: (64, 96, 96, 3)]. And then create a generator which is passed to our model while training. We use RMSProp optimizer and loss is triplet loss defined by function triplet_loss. After training save the model.

**predictions_and_finding_similar_images.py** - we predict and find similar images using Nearestneighbors with cosine distance. And plot them using matplotlib.

**Note:** This may not be final model since it is not performing well, due to submission data of this assignment, I have created a repository. I will work on this and update gradually. Thank You.

**Edited:** I added using-pre-trained-model.ipynb file, which uses pre-trained model. You can use any other pre-trained model as well.
