import numpy as np
import csv,sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model,model_from_json
from keras.preprocessing.image import ImageDataGenerator

def load_data(data_file):
    print("Loading data from "+data_file)
    data, data_count = [], 0
    with open(data_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            data.append(row)
            data_count+=1
    del(data[0])
    data_count-=1
    x_train = np.zeros((data_count*2, 48 , 48, 1), dtype = int)
    y_train = np.zeros((data_count*2, 1), dtype = int)
    for i in range(data_count):
        pixels = np.array(data[i][1].split(' '),int).reshape(48,48,1)
        x_train[2*i] = pixels
        f_pixels = np.fliplr(pixels)
        x_train[2*i+1] = f_pixels
        y_train[2*i] = data[i][0]
        y_train[2*i+1] = data[i][0]
    y_train = np_utils.to_categorical(y_train,7)
    
    (x_train,y_train) = (x_train/255,y_train)
    (x_val,y_val) = (x_train[-2000:],y_train[-2000:])
    (x_train,y_train)=(x_train[:-2000],y_train[:-2000])

    return (x_train,y_train),(x_val,y_val)


#load data
(x_train,y_train),(x_val,y_val)=load_data(sys.argv[1])
datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

datagen.fit(x_train)

#load model
#model = GoodNet()
#model = load_model('hw3_model.h5')
#model.summary()

# load json and create model

json_file = open('hw3_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.summary()
print("Loaded model from 'hw3_model.json'...")


#training
'''
model.compile(loss='categorical_crossentropy',optimizer="sgd",metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=32,epochs=60,validation_data=[x_val,y_val])
'''
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), 
                    steps_per_epoch = len(x_train)/32, 
                    epochs=200, 
                    validation_data=[x_val,y_val])

#set checkpoint
'''
filepath="kaggle-update.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(x_train,y_train,batch_size=128,epochs=100,validation_data=[x_val,y_val],callbacks=callbacks_list, verbose=1)
'''

#evaluation

score = model.evaluate(x_train,y_train)
print('\nTraining Acc:', score[1])
score = model.evaluate(x_val,y_val)
print('\nValidation Acc:', score[1])

#saving model
'''
print("Saving new trained model...")
model.save('new_trained_model.h5')
'''
