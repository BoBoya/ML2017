import csv,sys,os
import numpy as np
from scipy.special import expit

def readfile(filename,type,opt):
    data = []
    with open(filename,'rb') as f:
        if opt == 'pop':
            for row in csv.reader(f,delimiter=','):
                data.append(row)
            data.pop(0)
            data = np.array(data).astype(np.float)
            normalize(data)
        else:
            for row in csv.reader(f,delimiter=','):
                data.append(row[0])
            data = np.array(data).astype(np.int)
        if type == 'train':
            train_set = data[:32561]
            val_set = data[32561:]
            return train_set,val_set
        elif type == 'test':
            return data

def writefile(filename,Y):
    with open(filename,'w') as f:
        writer = csv.writer(f)
        writer.writerows([['id','label']])
        for i in range(len(Y)):
            writer.writerows([[str(i+1),Y[i]]])
def normalize(data):
    var = np.std(data,axis = 0)
    mean = np.sum(data,axis = 0)/len(data)
    index = [0,1,3,4,5]
    for i in index:
        data[:,i] = (data[:,i]-mean[i])/var[i]
    data[:,-1]=1

def sigmoid(z):
    res = np.divide(1,(float(1)+np.exp(-z)))
    return np.clip(res,0.0000000001,0.9999999999)

def logistic(X,Y):
    w= np.random.normal(0,1,106)
   # w= np.zeros(106)
    w[-1]=1.0
    a = 0.001
    itera = 5000
    lamda = 0.001
    beta_1 = 0.9
    beta_2 = 0.99
    epsilon = 10**(-8)
    m = np.zeros(106)
    m_hat = np.zeros(106)
    vec = np.zeros(106)
    vec_hat = np.zeros(106)
    loss = 0
    for i in range(itera):
        print "Iteration: %d    " % (i+1),
        pred = sigmoid(np.dot(X,w))
        diffy = np.subtract(pred,Y)
        loss = np.sum(np.negative(Y*np.log(pred)+(1-Y)*(np.log(1-pred))))
        #print "Loss: %f" % float(loss)
        g = np.dot(X.T, diffy) * 2 + lamda*2*w
        m = beta_1*m + (1 - beta_1)*g
        vec = beta_2*vec + (1 - beta_2)*(g**2)
    
        m_hat = m/(1 - beta_1**itera)
        vec_hat = vec/(1 - beta_2**itera)
        w = w - a*m_hat/(np.sqrt(vec_hat) + epsilon)
        print "Loss: %f\r" % float(loss),
    print "Final Loss: %f                " % float(loss)
    return w

def validation(X,Y,w):
    Y_p = predict(X,w)
    wrong = 0
    for i in range(len(Y)):
        if Y_p[i] != Y[i]:
            wrong += 1
    print "Accuracy:",1-np.divide(float(wrong),float(len(Y)))

def predict(Test,w):
    pred = sigmoid(np.dot(Test,w))
    out = []
    for i in range(len(pred)):
        if pred[i] >= 0.5:
            out.append(1)
        else:
            out.append(0)
    return out

if __name__ == '__main__':
    filename = []
    if len(sys.argv) != 7:
        print "wrong format"
        sys.exit()
    else:
        for i in range(1,7):
            filename.append(sys.argv[i])
    X_train,X_val = readfile(filename[2],'train','pop')
    Y_train,Y_val = readfile(filename[3],'train','normal')
    Test = readfile(filename[4],'test','pop')
    w = logistic(X_train,Y_train)
    #validation(X_val,Y_val,w)
    pred = predict(Test,w)
    writefile(filename[5],pred)
    with open('weight', 'w') as file:
        file.write('ggggggggg\n')
        for element in w:
            file.write(str(float(element))+'\n')
