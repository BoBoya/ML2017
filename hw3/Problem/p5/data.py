import numpy as np
import csv,sys
from keras.utils import np_utils
from keras import backend as K
from timeit import default_timer as timer
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model,model_from_json
from keras.optimizers import SGD, Adam
def load_data(file_):
    data, data_count = [], 0
    test, test_count = [], 0
    with open(file_, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            data.append(row)
            data_count+=1
    del(data[0])
    data_count-=1
    test_count-=1
    x_train= np.zeros((data_count, 48 , 48, 1), dtype = int)
    y_train= np.zeros((data_count, 1), dtype = int)
    for i in range(data_count):
        pixels = np.array(data[i][1].split(' '),int).reshape(48,48,1)
        x_train[i] = pixels
        #f_pixels = np.fliplr(pixels)
        #x_train[2*i+1] = f_pixels
        y_train[i] = data[i][0]
        #y_train[2*i+1] = data[i][0]
    y_train = np_utils.to_categorical(y_train,7)
    
    (x_train,y_train)=(x_train/255,y_train)
    (x_val,y_val) = (x_train[-3709:],y_train[-3709:])
    (x_train,y_train)=(x_train[:-3709],y_train[:-3709])

    return (x_val,y_val)
    

def training(x_train,y_train,x_val,y_val):
    #K.set_image_dim_ordering("th")
    datagen = ImageDataGenerator(
    #shear_range = 0.1,
    #rescale = 1./255,
    zoom_range=0,
    rotation_range=20,  
    width_shift_range=0.1,  
	height_shift_range=0.1,  
	horizontal_flip=True, 
	vertical_flip=False)
    datagen.fit(train_X)

    model = AlexNet()
    model.compile(loss='categorical_crossentropy',optimizer="sgd",metrics=['accuracy'])
    #model.fit(x_train,y_train,batch_size=32,epochs=64,validation_data=[x_val,y_val])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),steps_per_epoch = len(x_train)/32, epochs=100, validation_data=[x_val,y_val])
    score = model.evaluate(x_train,y_train)
    print('\nTraining Acc:', score[1])
    score = model.evaluate(x_val,y_val)
    print('\nValidation Acc:', score[1])
    model.save('models/datagen-b32-100.h5')
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=64),steps_per_epoch = len(x_train)/32, epochs=50, validation_data=[x_val,y_val])
    score = model.evaluate(x_train,y_train)
    print('\nTraining Acc:', score[1])
    score = model.evaluate(x_val,y_val)
    print('\nValidation Acc:', score[1])
    model.save('models/datagen-b64-50.h5')

if __name__ == '__main__':
    (x_train,y_train),(x_test,y_test),(x_val,y_val)=load_data()
    training(x_train,y_train,x_val,y_val)
    
