# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:01:29 2017

@author: imtiaz.a.khan
"""

'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import os
os.chdir('C:/Users/imtiaz.a.khan/Desktop/Kaggle/MNIST')

batch_size = 128
num_classes = 10
epochs = 20

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#Creating train and validation sets
df_train, df_valid = train_test_split(train,               # Data set to split
                                   test_size = 0.20,                # Split ratio
                                   random_state=100,    # Set random seed
                                   stratify = train['label'])

x_train=df_train.iloc[:,1:]
x_valid=df_valid.iloc[:,1:]
# the data, shuffled and split between train and test sets


x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_train /= 255
x_valid /= 255
x_train = np.array(x_train)
x_train = x_train.reshape(33600, 784)
x_valid = np.array(x_valid)
x_valid = x_valid.reshape(8400, 784)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'validation samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(df_train.label, num_classes)
y_valid = keras.utils.to_categorical(df_valid.label, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_valid, y_valid))
score = model.evaluate(x_valid, y_valid, verbose=0)


print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict_classes(np.array(test).reshape(28000,784), verbose=0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv('digitRecogniserKaggleSubmission.csv',index=False)
