from keras.models import load_model
from sklearn.metrics import confusion_matrix
#from marcos import exp_dir
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
import csv,sys
from keras.utils import np_utils

def load_data(file_):
    data, data_count = [], 0
    with open(file_, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            data.append(row)
            data_count+=1
    del(data[0])
    data_count-=1
    x_train= np.zeros((data_count, 48 , 48, 1), dtype = int)
    y_train= np.zeros((data_count, 1), dtype = int)
    for i in range(data_count):
        pixels = np.array(data[i][1].split(' '),int).reshape(48,48,1)
        x_train[i] = pixels
        #f_pixels = np.fliplr(pixels)
        #x_train[2*i+1] = f_pixels
        y_train[i] = data[i][0]
        #y_train[i] = 0
        #y_train[2*i+1] = data[i][0]
    y_train = np_utils.to_categorical(y_train,7)
    
    (x_train,y_train)=(x_train/255,y_train)
    (x_val,y_val) = (x_train[-3709:],y_train[-3709:])
    (x_train,y_train)=(x_train[:-3709],y_train[:-3709])

    return (x_val,y_val)
def plot_confusion_matrix(cm, classes,
                title='Confusion matrix',
                cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    #model_path = os.path.join(exp_dir,store_path,'model.h5')
    print ("loading model")
    emotion_classifier = load_model('../hw3_model.h5')
    np.set_printoptions(precision=2)
    (x_val,y_val)=load_data(sys.argv[1])
    dev_feats = x_val
    predictions = emotion_classifier.predict_classes(dev_feats)
    te_labels = np.argmax(y_val,axis=1)
    #print (predictions,te_labels)
    conf_mat = confusion_matrix(te_labels,predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    plt.savefig('confusion_matrix.png')
    #plt.show()
if __name__ == '__main__':
    matplotlib.use('Agg')
    main()
