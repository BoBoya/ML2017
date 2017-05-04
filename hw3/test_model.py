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
from alexnet import AlexNet
#from cifar10 import cifar10

def load_data(test_file):
    print("Loading data from "+test_file)
    test, test_count = [], 0
    with open(test_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            test.append(row)
            test_count+=1
    del(test[0])
    test_count-=1
    x_test = np.zeros((test_count, 48, 48, 1), dtype = int)
    for i in range(test_count):
        pixels = np.array(test[i][1].split(' '),int).reshape(48,48,1)
        x_test[i] = pixels
    
    (x_test)=(x_test/255)

    return (x_test)



#load model
model = load_model('hw3_model.h5')
model.compile(loss='categorical_crossentropy',optimizer="sgd",metrics=['accuracy'])
model.summary()

#load data
(x_test) = load_data(sys.argv[1])

#predict
print("Start predicting...")
pred = model.predict_classes(x_test)
with open(sys.argv[2],'w') as output:
    print("Writing prediction to "+sys.argv[2]+"...")
    writer = csv.writer(output,delimiter=',')
    writer.writerow(['id','label'])
    d = 0
    for element in pred:
        writer.writerow([str(d),str(int(element))])
        d+=1

