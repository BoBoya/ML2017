from PIL import Image
from glob import glob
import numpy as np
import os
import matplotlib.pyplot as plt
image_list = []

for char in range(65,75):
    for num in range(10):
        im = Image.open('../data/'+chr(char)+'0'+str(num)+'.bmp')
        pix = list(im.getdata())
        image_list.append(pix)

X = np.array(image_list)
print (X.shape)
X_mean = X.mean(axis=0, keepdims=True)
X_ctr = X - X_mean
u, s, v = np.linalg.svd(X_ctr)
print (v.shape)

eigen = np.zeros((5, 64*64))

for i in range(0, 5):
    eigen[i] = v[i]

eigen_T = eigen.T

coefficient = np.dot(X_ctr, eigen_T)
recon = np.zeros((100, 4096))
recon = np.dot(coefficient, eigen) + X_mean

"""
plt.figure(1)
for i in range(1,10):
    vi = np.array(v[i])
    vi = vi.reshape(64,64)
    plt.subplot('33{}'.format(i))
    plt.imshow(vi,cmap='gray')

plt.savefig('{}.png'.format('total_'))

plt.figure(2)
for i in range(100):
    plt.subplot(10,10,(i+1))
    f = plt.gca()
    f.axes.get_xaxis().set_visible(False)
    f.axes.get_yaxis().set_visible(False)
    plt.imshow(recon[i].reshape(64,64),cmap='gray')
plt.savefig('{}.png'.format('recover'))

plt.figure(3)
for i in range(0, 100):
    plt.subplot(10,10,i+1)
    f = plt.gca()
    f.axes.get_xaxis().set_visible(False)
    f.axes.get_yaxis().set_visible(False)
    plt.imshow(X[i].reshape(64,64), cmap='gray')
plt.savefig('original.png')
"""
for i in range(1, 100):
    
    eigen = v[0:i]
    eigen_T = eigen.T
    
    coefficient = np.dot(X_ctr, eigen_T)
    recon = np.dot(coefficient, eigen) + X_mean
    diff = recon - X
    loss = np.sqrt(1/100 * 1/4096 * np.trace(np.dot(diff, diff.T)))/256
    
    if loss < 0.01:
        print('loss: {:f}\r'.format(loss))
        print(i)
        break
    else:
        print('loss: {:f}\r'.format(loss), end = '')
