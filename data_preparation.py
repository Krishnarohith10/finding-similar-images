"""
In this python file we are changing the size of an image from 512x512 to 96x96 since 
our created model will take input image size of 96x96. '.\\path_dict.pickle' is a 
pickle file created where it contains the paths of classes of our dataset_old. This 
is just data preparation so the name 'data_preparation'.
"""
import os
import cv2
import pickle

with open('.\\path_dict.pickle', 'rb') as f:
    paths = pickle.load(f)

for key in paths.keys():
    for image in os.listdir(paths[key]):
        img = cv2.imread(os.path.join(paths[key], image))
        img = cv2.resize(img, (226, 226), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(paths[key], image), img)
    print('Completed! Key:', key)
