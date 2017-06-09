import sys
import os
import pandas as pd
import numpy as np
import pickle
import keras.backend as K

from keras.models import Sequential, Model
from keras.layers import Merge, Dense, Dropout, Activation, Flatten, Concatenate, Input, Dot, Add
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
from keras.regularizers import l1, l2
import csv


NB_EPOCHS = 150
BATCH_SIZE = 2048
SPLIT_RATIO = 0.1
LATENT_DIM = 16
n_users = 6041
n_movies = 3953

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
def load_data():
    print('Loading data.')
    test = np.genfromtxt(sys.argv[1]+'test.csv', delimiter=",", dtype=int)[1:]
    return test

def normalize(data):
    print('normalizing data')
    f_data = np.array(data,dtype=np.float_)
    r_mean = np.mean(data[:,3])
    r_std = np.std(data[:,3])
    nor_rating = np.divide(f_data[:,3]-r_mean,r_std)
    return nor_rating, r_mean, r_std

def MF_model(n_users, n_items, latent_dim):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users,latent_dim,embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items,latent_dim,embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(n_users,1,embeddings_initializer = 'zeros')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items,1,embeddings_initializer = 'zeros')(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vec, item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = Model([user_input, item_input], r_hat)
    model.compile(loss='mse',optimizer='rmsprop')
    return model


def DNN_model(n_users, n_items, latent_dim):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users,latent_dim,embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items,latent_dim,embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    merge_vec = Concatenate()([user_vec,item_vec])
    hidden = Dense(256,activation='relu')(merge_vec)
    hidden = Dropout(0.5)(hidden)
    #hidden = Dense(32,activation='relu')(hidden)
    output = Dense(1)(hidden)
    model = Model([user_input, item_input], output)
    model.compile(loss='mse',optimizer='rmsprop')
    model.summary()
    return model

def training(normalization,use_DNN):
    '''
    n_users = int(np.max(list(set(train[:,1])))+1)
    n_movies = int(np.max(list(set(train[:,2])))+1)

    if normalization:
        nor_rating, r_mean, r_std = normalize(train)

    print('Splitting data.')
    VALID = int(SPLIT_RATIO*len(train))
    valid = train[:VALID]
    train = train[VALID:]
    valid_gt = valid[:,3]
    train_gt = train[:,3]

    if normalization:
        valid_gt = nor_rating[:VALID]
        train_gt = nor_rating[VALID:]
    '''  
    model = MF_model(n_users, n_movies,LATENT_DIM)
    if use_DNN:
        model = DNN_model(n_users, n_movies, LATENT_DIM)
    '''
    earlystopping = EarlyStopping(monitor='val_loss', patience = 5, verbose=1)#, mode='min')
    
    checkpoint = ModelCheckpoint(filepath='weights/MF_best.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 mode='min')
    
    model.fit([train[:,1], train[:,2]], train_gt,
                        validation_data=([valid[:,1], valid[:,2]], valid_gt),
                        batch_size=BATCH_SIZE,
                        epochs=NB_EPOCHS,
                        callbacks=[earlystopping,checkpoint])
    '''
    print("Loading best model")
    model.load_weights('MF_best.hdf5')
    '''
    y_true = train[:,3]
    y_pred = model.predict([train[:,1], train[:,2]]).flatten()
    if normalization:
        y_pred = y_pred * r_std + r_mean
    rmse = np.sqrt(np.mean(np.square(y_true-y_pred)))
    print('RMSE on training data', rmse)
    y_true = valid[:,3]
    y_pred = model.predict([valid[:,1], valid[:,2]]).flatten()
    if normalization:
        y_pred = y_pred * r_std + r_mean
    rmse = np.sqrt(np.mean(np.square(y_true-y_pred)))
    print('RMSE on validation data', rmse)
    '''

    if normalization:
        return model,r_mean,r_std
    else:
        return model


def write_data(data):
    f = open(sys.argv[2],"w")
    w = csv.writer(f)
    w.writerow(["TestDataID","Rating"])
    w.writerows(data)
    f.close()

if __name__ == '__main__':
    
    test = load_data()
    model = training(False,False)
    testID = test[:,0]
    y_test = model.predict([test[:,1], test[:,2]]).flatten()
    result = []
    for i in range(len(y_test)):
        result.append([testID[i], y_test[i]])
    write_data(result)
    
