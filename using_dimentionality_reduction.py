import os
import cv2
import pickle

with open('path_dict.pickle', 'rb') as f:
    paths = pickle.load(f)

import numpy as np
data_orig = []
data_tsne = []
for key in paths.keys():
    length=0
    for image in os.listdir(paths[key]):
        img = cv2.imread(os.path.join(paths[key], image))
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        data_orig.append(img)
        img = img.reshape(-1)
        data_tsne.append(img)
        length += 1
    print('Completed key:', key, 'Length of images:', length)

del paths, length, img, image, key, f

data_orig = np.array(data_orig)
data_tsne = np.array(data_tsne)
data_tsne = data_tsne/255

from sklearn.model_selection import train_test_split
x_train_orig, x_test_orig = train_test_split(data_orig, test_size=0.02, shuffle=True, random_state=24)

del data_orig

from sklearn.decomposition import PCA
print('creating PCA with n_components=100........', end='')
pca = PCA(n_components=100, svd_solver='randomized', random_state=4)
print('completed!')
print('fitting on data..........', end='')
pca.fit(data_tsne)
data_tsne_pca = pca.transform(data_tsne)
print('completed!')

del data_tsne

from sklearn.manifold import TSNE
print('creating t-SNE algorithm..........', end='')
tsne = TSNE(n_components=2, metric='cosine')
print('created!')
print('fitting on data.........', end='')
data_embeddings = tsne.fit_transform(data_tsne_pca)
print('completed!')

del data_tsne_pca

x_train_embeddings, x_test_embeddings = train_test_split(data_embeddings, test_size=0.02, shuffle=True, random_state=24)

del data_embeddings

from sklearn.neighbors import NearestNeighbors
print('creating nearest neighbors with n_neighbors=5...........', end='')
nn = NearestNeighbors(n_neighbors=5, metric='cosine')
print('created!')
print('fitting on t-SNE data........', end='')
nn.fit(x_train_embeddings)
print('completed!')

print('Finding similar images using NearestNeighbors for random image in test data')
num = np.random.randint(len(x_test_embeddings))
indices = nn.kneighbors([x_test_embeddings[num]], return_distance=False)
img_query = x_test_orig[num]
img_retrieval = [x_train_orig[idx] for idx in indices]
print('Completed!')

import matplotlib.pyplot as plt
print('Plotting query image with similar images........', end='')
plt.figure()
plt.imshow(cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB))
plt.figure()
for i, img in enumerate(img_retrieval[0]):
    plt.subplot(1,5,i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
print('completed!')
