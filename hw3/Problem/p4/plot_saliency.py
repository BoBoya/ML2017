#!/usr/bin/env python
# -- coding: utf-8 --

import os
import sys
import argparse
from sklearn.preprocessing import normalize
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
from data import load_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
'''
base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
model_dir = os.path.join(base_dir, 'model')
'''
def compute_heatmap(hm):
    print('heatmap min: ',hm.min())
    print('heatmap max: ',hm.max())
    heatmap = (hm-hm.min())/(hm.max()-hm.min())
    thres = heatmap.mean()+heatmap.std()*0.20
    #thres = heatmap.mean()
    print ('threshold: ',thres)
    print ('active ratio: ',float(len(heatmap[np.where(heatmap >= thres)])/(48*48)))
    heatmap[np.where(heatmap <= thres)] = thres*0.999
    heatmap = heatmap.reshape((48,48))

    return (heatmap,thres)
    


def main():
    #parser = argparse.ArgumentParser(prog='plot_saliency.py',
    #description='ML-Assignment3 visualize attention heat map.')
    #parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=80)
    #args = parser.parse_args()
    model_name = "../hw3_model.h5" 
    #model_path = os.path.join(model_dir, model_name)
    emotion_classifier = load_model(model_name)
    print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))

    private_pixels,y_val = load_data(sys.argv[1])
    private_pixels = [ private_pixels[i].reshape((1, 48, 48, 1)) 
                        for i in range(len(private_pixels)) ]
    input_img = emotion_classifier.input
    img_ids =  range(1)
    #prediction = emotion_classifier.predict(private_pixels)

    for idx in img_ids:
        val_proba = emotion_classifier.predict(private_pixels[idx])
        #val_proba = prediction[idx]
        
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])
        
        out = fn([private_pixels[idx],0])[0]
        hm = np.array(out)
        (heatmap,thres)=compute_heatmap(hm)
        '''
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        '''
        

        see = private_pixels[idx].reshape(48, 48)
        see[np.where(heatmap <= thres)] = np.mean(see)

        print('saving heatmap_'+str(idx))
        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('heatmap_'+str(idx)+'.png', dpi=100)

        print('saving mask_'+str(idx))
        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('mask_'+str(idx)+'.png', dpi=100)

if __name__ == "__main__":
    matplotlib.use('Agg')
    main()
