#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:12:29 2018

@author: alan
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import coremltools
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from keras.layers import Dense, Activation


train = pd.read_csv("jsonformatter.csv")
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
shape = X_train.shape
model = Sequential()
model.add(Dense(32, input_dim=3))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(13))

model.add(Activation('softmax'))
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
epochs = 300 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 32

model.fit(X_train,Y_train,epochs=150,batch_size=32)
print(model.summary())

coreml_device = coremltools.converters.keras.convert(model)
print(coreml_device)

coreml_device.save('gravitytoclockposition.mlmodel')