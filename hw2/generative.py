import numpy as np
import csv,sys,os
from logistic import readfile
from logistic import writefile
from logistic import sigmoid
dim = 106

def generative(X,Y):
    print "Computing Guassian Distribution Parameters..."
    size = X.shape[0]
    num1 = np.count_nonzero(Y == 1)
    num0 = np.count_nonzero(Y == 0)
    m1 = np.zeros((dim,))
    m0 = np.zeros((dim,))
    sig1 = np.zeros((dim,dim))
    sig0 = np.zeros((dim,dim))
    for i in range(size):
        if Y[i] == 1:
            m1 += X[i]
        else:
            m0 += X[i]
    m1 /= num1
    m0 /= num0
    for i in range(size):
        if Y[i] == 1:
            sig1 += np.dot(np.transpose([X[i] - m1]), [(X[i] - m1)])
        else:
            sig0 += np.dot(np.transpose([X[i] - m0]), [(X[i] - m0)])
    sig1 /= num1
    sig0 /= num0
    s_sig = (float(num1)/size)*sig1 + (float(num0)/size)*sig0
    print "Shared sigma:\n",s_sig
    return (m1,m0,s_sig,num1,num0)

def predict(X_test,m1,m0,sigma,n1,n0):
    print "Predicting..."
    sig_inv = np.linalg.inv(sigma)
    x = X_test.T
    w = np.dot((m1-m0),sig_inv)
    b = (-0.5) * np.dot(np.dot([m1], sig_inv), m1) + (0.5) * np.dot(np.dot([m0], sig_inv), m0) + np.log(float(n1)/n0)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    out = []
    for i in range(len(y)):
        if y[i] >= 0.5:
            out.append(1)
        else:
            out.append(0)
    return out

if __name__ == "__main__":
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
    m1,m0,sig,n1,n0 = generative(X_train,Y_train)
    Y_pred = predict(Test,m1,m0,sig,n1,n0)
    writefile(filename[5],Y_pred)
