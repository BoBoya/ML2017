import csv,sys
import numpy as np
from logistic import writefile
from logistic import readfile
from logistic import predict

if __name__ == '__main__':
    filename = []
    if len(sys.argv) != 7:
        print "wrong format"
        sys.exit()
    else:
        for i in range(1,7):
            filename.append(sys.argv[i])
    Test = readfile(filename[4],'test','pop')
    w = []
    with open('weight','rb') as f:
        for row in f:
            w.append(row[:-1])
    w.pop(0)
    w = np.array(w).astype(np.float)
    print "\n"
    print "I forget to record the parameters which I reach my highest public score."
    print "This program record the weight that reach the public score 0.85332"
    print "Weight:"
    print w
    print "\n"
    pred = predict(Test,w)
    writefile(filename[5],pred)

