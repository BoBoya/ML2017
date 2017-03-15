import csv
import numpy as np

def gd_real(matrix,ans,option,a,numI):
    w = np.ones([len(option)*10])
    #w = np.random.normal(0,1,[len(option)*10])
    print w
    for i in range(500):
        loss = float(0)
        for j in range(240):
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
            loss = np.sqrt(np.sum((h-answer)**2)/6)+np.sum(0.00001*(w**2))
            gradient = (2*np.dot((h - answer),data))/6+0.00001*2*w
            w = w - a * gradient
        print "Iteration:",i," Loss:",loss

    return w 

def gd(matrix,ans,w,a,numI):
    for i in range(100):
        for j in range(240):
            loss = float(0)
            gradient = float(0)
            inform = np.array(matrix['PM2.5'][j*6:(j+1)*6])
            answer = ans[j*6:(j+1)*6]
            h = np.dot(inform,w)
            loss = np.sqrt(np.sum((h-answer)**2)/6)+np.sum(0.00001*w**2)
            gradient = (2*np.dot((h - answer),inform))/6+0.00001*2*w
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

def test(t_type,data,matrix):
    Y = []
    loss = 0
    for i in range(len(data['RAINFALL'])):
        test = []
        for opt in option:
            if t_type =='train':
                test.extend(matrix[opt][i])
            elif t_type == 'test':
                test.extend(data[opt][i])
        test = np.array(test)
        predict = np.dot(weight,test)
        if t_type == 'train':
            print predict, ans[i]
            loss += np.abs(predict-ans[i])
        elif t_type == 'test': 
            Y.append(predict)
    if t_type == 'train':
        print (loss/len(data['PM2.5']))
    else:
        return Y

if __name__=='__main__':
    matrix,ans = readfile('train')
    option = ['PM2.5']
    #weight = np.random.normal(0, 1, 10)
    weight = np.ones(10)
    weight = gd(matrix,ans,weight,1e-5,1)
    #weight = gd_real(matrix,ans,option,1e-6,1)
    print 'final weight: \n',weight
    data = readfile('test')
    test('train',data,matrix)
    writefile('out2.csv',test('test',data,matrix))
