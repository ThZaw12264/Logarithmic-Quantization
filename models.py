import numpy as np
import pandas as pd

#hey der
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import imageio

from fastai import *
from fastai.vision import *
from fastai.vision.all import *


df = pd.read_csv('../input/clothing-single-channel/fashion-mnist_train.csv')

df_x = df.loc[:,'pixel0':'pixel783']
df_y = df.loc[:,'label']

np_x = np.array(df_x)

X_train = np.array(np_x).reshape(-1,28, 28)
y_train = np.array(df_y)

print(X_train.shape)
print(y_train.shape)

X_train = np.stack((X_train,)*3, axis=-1)
print(X_train.shape)

def save_imgs(path:Path, data, labels):
    for label in np.unique(labels):
        (path/str(label)).mkdir(parents=True,exist_ok=True)
    for i in range(len(data)):
        if(len(labels)!=0):
            imageio.imsave( str( path/str(labels[i])/(str(i)+'.jpg') ), data[i])
        else:
            imageio.imsave( str( path/(str(i)+'.jpg') ), data[i])

save_imgs(Path('/data/digits'),X_train,y_train)

from keras.applications.resnet50 import preprocess_input
import tensorflow as tf

print('total classes :', len(os.listdir('/data/digits')))
print('Images with label 1: ', len(os.listdir('/data/digits/1')))

print('Image names with label 1')
print(os.listdir('/data/digits/1')[:10])

train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.15)
train_generator = train_datagen.flow_from_directory('/data/digits', class_mode='categorical', subset='training')
valid_generator = train_datagen.flow_from_directory('/data/digits', class_mode='categorical', subset='validation')

x, y = train_generator[0]
print(x.shape)
print(y.shape)

resnet_weights_path = '../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

from tensorflow.keras.applications.resnet50 import ResNet50
model = Sequential()

model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))

# 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
model.add(Dense(10, activation = 'softmax'))

# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = False

model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(generator=train_generator,epochs=10,validation_data=valid_generator)
