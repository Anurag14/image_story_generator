'''script for I/O -->clustering -->generating story --sklearn'''
import numpy as np
import os
import glob
from sklearn.cluster import KMeans
# resize an image using the PIL image library

import PIL
from PIL import Image
src_dir = input("Please enter your absolute source image directory, all images in *.jpg format\n")

width,height=28,28
ds=[]
for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
    im1 = Image.open(jpgfile)
    im5=im1.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
    ds.append(np.array(img))
data=np.array(ds)    


""" now we have the data as an numpy array in data. We then feed this to the autoencoder
    and then iteratively append all the encoded features into the numpy array X
"""

from keras.models import Model
from keras.models import load_model
import time
print('Loading model :')
t0 = time.time()
autoencoder = load_model('autoencoder.h5')
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
t1 = time.time()
print('Model loaded in: ', t1-t0)
data = x_train.astype('float32') / 255.
data = np.reshape(data, (len(data), 28, 28, 1))  # adapt this if using `channels_first` image data format
X=encoder.predict(data)
X = X.reshape(X.shape[0],X.shape[1] * X.shape[2] * X.shape[3])
#now clustering the input 


cluster=int(input("How many clusters do you want?\n"))
kmeans=KMeans(n_clusters=cluster)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
print(kmeans.labels_)
dictionary={cluster_label: np.where(kmeans.labels_ == cluster_label)[0] for cluster_label in range(kmeans.n_clusters)}
dx=[]
for cluster_label in range(cluster):
    for item in dictionary[cluster_label]:
        if np.array_equal(X[item],centroids[cluster_label]):
            dx.append(item)
            break
dx.sort()

#now we save the images in a destination folder corresponding to indices in the list dx
import shutil
dest_dir = input("Please enter your absolute dest image directory, all images in *.jpg format\n")
counter=0
for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
    if(counter>len(dx)):
        break
    if(counter==dx[i]):
        i+=1
        shutil.copy(jpgfile, dst_dir)
    counter+=1
