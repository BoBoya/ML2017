import csv
import numpy as np
import matplotlib.pyplot as plt
def gd_real(matrix,ans,option,a,numI,valid,lamd):
    w = np.ones([len(option)*10])
    #w = np.random.normal(0,1,[len(option)*10])
    #print w
    for i in range(500):
        loss = float(0)
        for j in range(valid):
            gradient = float(0)
            inform =[]
            for k in option:
                inform.append(np.array(matrix[k][j*6:(j+1)*6]))
            answer = ans[j*6:(j+1)*6]
            data = []
            for t1 in range(6):
                line = []
                for t2 in range(len(option)):
                    line.extend(inform[t2][t1])
                data.append(line)
            data = np.array(data)
            h = np.dot(data,w)
            loss = np.sqrt(np.sum((h-answer)**2)/6)+np.sum(lamd*(w**2))
            gradient = (2*np.dot((h - answer),data))/6+lamd*2*w
            w = w - a * gradient
        #print "Iteration:",i," Loss:",loss

    return w 

def gd(matrix,ans,w,a,numI,valid,lamd):
    for i in range(100):
        for j in range(valid):
            loss = float(0)
            gradient = float(0)
            inform = np.array(matrix['PM2.5'][j*6:(j+1)*6])
            answer = ans[j*6:(j+1)*6]
            h = np.dot(inform,w)
            loss = np.sqrt(np.sum((h-answer)**2)/6)+np.sum(lamd*w**2)
            gradient = (2*np.dot((h - answer),inform))/6+lamd*2*w
            w = w - a * gradient
            #b = b - 2 * a *(h - ans[j])
            #print "Iter:",i,"; Batch:",j,"; Loss: ",loss
    return w

def readfile(type):
    matrix = dict()
    if type =='train':
        f = open('train.csv','r')
        ans = []
        read = False
        for row in csv.reader(f):
            if read == True:
                if row[2] not in matrix:
                    matrix[row[2]]=[]
                for i in range(6):
                    if row[2] == 'RAINFALL':
                        data = []
                        for j in row[3+i:12+i]:
                            if j == 'NR':
                                data.append(float(0))
                            else:
                                data.append(float(j))
                    elif row[2] == 'WIND_SPEED'or row[2] == 'SO2' or row[2]== 'THC':
                        data = [float(j)*10 for j in row[3+i:12+i]]
                    else:
                        data = [float(j) for j in row[3+i:12+i]]
                    data.extend([1])
                    matrix[row[2]].append(data)
                    if row[2] == 'PM2.5':
                        ans.append(float(row[12+i]))
            read = True
        return matrix,ans

    elif type == 'test':
        f = open('test_X.csv','r')
        for row in csv.reader(f):
            if row[1] not in matrix:
                matrix[row[1]]=[]
            if row[1] == 'RAINFALL':
                data = []
                for j in row[2:]:
                    if j == 'NR':
                        data.append(float(0))
                    else:
                        data.append(float(j))
            elif row[1] == 'WIND_SPEED' or row[1]== 'SO2' or row[1]== 'THC':
                data = [float(j)*10 for j in row[2:]]
            else:
                data = [float(j) for j in row[2:]]
            data.extend([1])
            matrix[row[1]].append(data)
        return matrix

def writefile(fileName, Y):
    with open(fileName, 'w') as f:
        writer = csv.writer(f)
        writer.writerows([['id','value']])
        for i in range(len(Y)):
            writer.writerows([['id_'+str(i),Y[i]]])

def test(t_type,data,matrix,option):
    Y = []
    loss = 0
    if t_type == 'train':
        base = 200
    else:
        base = 0
    for i in range(base,len(data['RAINFALL'])):
        test = []
        for opt in option:
            if t_type =='train':
                test.extend(matrix[opt][i])
            elif t_type == 'test':
                test.extend(data[opt][i])
        test = np.array(test)
        predict = np.dot(weight,test)
        if t_type == 'train':
#            print predict, ans[i]
            loss += np.square(predict-ans[i])
            #loss += np.abs(predict-ans[i])
        elif t_type == 'test': 
            Y.append(predict)
    if t_type == 'train':
        Loss =np.sqrt(loss/(len(data['PM2.5'])-base))
        print "Loss:" ,np.sqrt(loss/(len(data['PM2.5'])-base))
        return Loss 
    else:
        return Y

if __name__=='__main__':
    matrix,ans = readfile('train')
    option = ['PM2.5']
    #weight = np.random.normal(0, 1, 10)
    weight = np.ones(10)
    weight = gd(matrix,ans,weight,1e-5,1,240,0)
    #weight = gd_real(matrix,ans,option,1e-6,1)
    print 'final weight: \n',weight
    data = readfile('test')
    test('train',data,matrix,option)
    writefile('output.csv',test('test',data,matrix,option))

    #This part is for the report
    '''
    A=[]
    B=[]
    C=[]
    D=[]
    for i in range(20):
        print "Training ",(i+1)*10," data"
        weight = np.ones(10)
        weight = gd(matrix,ans,weight,1e-5,1,(i+1)*10,0)
        option=['PM2.5']
        Loss=test('train',data,matrix,option)
        A.append([(i+1)*10,Loss])
        weight = gd(matrix,ans,weight,1e-5,1,(i+1)*10,0.0001)
        option=['PM2.5']
        Loss=test('train',data,matrix,option)
        A.append([(i+1)*10,Loss])
        weight = gd(matrix,ans,weight,1e-5,1,(i+1)*10,0.00001)
        Loss=test('train',data,matrix,option)
        B.append([(i+1)*10,Loss])
        weight = gd(matrix,ans,weight,1e-5,1,(i+1)*10,0.000001)
        Loss=test('train',data,matrix,option)
        C.append([(i+1)*10,Loss])
        
        option =['PM2.5','PM10']
        weight = gd_real(matrix,ans,option,1e-5,1,(i+1)*10,0)
        Loss=test('train',data,matrix,option)
        B.append([(i+1)*10,Loss])
        option =['PM2.5','PM10','NO2']
        weight = gd_real(matrix,ans,option,1e-5,1,(i+1)*10,0)
        Loss=test('train',data,matrix,option)
        C.append([(i+1)*10,Loss])
    A=np.array(A)
    B=np.array(B)
    C=np.array(C)
    #D=np.array(D)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.legend(loc='lower left')
    plt.title(u"Training Data-Loss")
    plt.xlabel(u"Training Data(10days)")
    plt.ylabel(u"Loss")
    plt.plot(range(1,20), A[1:, 1], label=u'PM2.5')
    #plt.plot(range(1,20), B[1:, 1], label=u'PM2.5+PM10')
    #plt.plot(range(1,20), C[1:, 1], label=u'PM2.5+PM10+NO2')
    #plt.plot(range(1,20), D[1:, 1], label=u'lamd=0')
    plt.legend(loc='upper right')
    plt.show()
    '''
        
