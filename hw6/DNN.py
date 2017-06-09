# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import numpy as np
import pickle
import keras.backend as K

from keras.models import Sequential, Model, load_model
from keras.layers import Merge, Dense, Dropout, Activation, Flatten, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
from keras.regularizers import l1, l2
import csv

NB_EPOCHS = 40
BATCH_SIZE = 2048
SPLIT_RATIO = 0.1
LATENT_FACTOR = 128
n_users = 6041
n_movies = 3953

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_data():
    print('Loading data.')
    test = np.genfromtxt(sys.argv[1]+'test.csv', delimiter=",", dtype=int)[1:]
    return test

def get_DNN():
    user_branch = Sequential()
    user_branch.add(Embedding(n_users, LATENT_FACTOR, embeddings_regularizer=l2(0.0), input_length=1))
    user_branch.add(Flatten())
    user_branch.add(BatchNormalization())

    movie_branch = Sequential()
    movie_branch.add(Embedding(n_movies, LATENT_FACTOR, embeddings_regularizer=l2(0.0), input_length=1))
    movie_branch.add(Flatten())
    movie_branch.add(BatchNormalization())

    model = Sequential()
    model.add(Merge([user_branch, movie_branch], mode = 'concat'))
    model.add(Dense(LATENT_FACTOR, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.load_weights('DNN_best.h5')
    return model
    

def train_model(train):
    n_users = np.max(list(set(train[:,1])))+1
    n_movies = np.max(list(set(train[:,2])))+1
    print(n_users,n_movies)

    print('Splitting data.')
    VALID = int(SPLIT_RATIO*len(train))
    valid = train[:VALID]
    train = train[VALID:]
    
    
    print('Training model.')
 
    user_branch = Sequential()
    user_branch.add(Embedding(n_users, LATENT_FACTOR, embeddings_regularizer=l2(0.0), input_length=1))
    user_branch.add(Flatten())
    user_branch.add(BatchNormalization())

    movie_branch = Sequential()
    movie_branch.add(Embedding(n_movies, LATENT_FACTOR, embeddings_regularizer=l2(0.0), input_length=1))
    movie_branch.add(Flatten())
    movie_branch.add(BatchNormalization())

    model = Sequential()
    model.add(Merge([user_branch, movie_branch], mode = 'concat'))
    model.add(Dense(LATENT_FACTOR, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.load_weights('DNN_best.h5')
    '''
    model.fit([train[:,1], train[:,2]], train[:,3],
                        validation_data=([valid[:,1], valid[:,2]], valid[:,3]),
                        batch_size=BATCH_SIZE,
                        epochs=NB_EPOCHS)
    model.save('DNN_best.h5')
    
    y_true = train[:,3]
    y_pred = model.predict([train[:,1], train[:,2]]).flatten()
    rmse = np.sqrt(np.mean(np.square(y_true-y_pred)))
    print('RMSE on training data', rmse)
    
    print('Loading best DNN model...')
    model = load_model('DNN_adam_model.h5')
    model.summary()
    '''
    y_true = valid[:,3]
    y_pred = model.predict([valid[:,1], valid[:,2]]).flatten()
    rmse = np.sqrt(np.mean(np.square(y_true-y_pred)))
    print('RMSE on validation data', rmse)
    
    return model
    

def write_data(data):
    f = open(sys.argv[2],"w")
    w = csv.writer(f)
    w.writerow(["TestDataID","Rating"])
    w.writerows(data)
    f.close()

if __name__ == '__main__':
    
    test = load_data()
    #model = train_model(train)
    model = get_DNN()
    testID = test[:,0]
    y_test = model.predict([test[:,1], test[:,2]]).flatten()
    result = []
    for i in range(len(y_test)):
        result.append([testID[i], y_test[i]])
    write_data(result)
