#!/usr/bin/env python
# -- coding: utf-8 --

import os,sys
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from data import load_data
import numpy as np

def main():
    emotion_classifier = load_model('../hw3_model.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    print (layer_dict)

    input_img = emotion_classifier.input
    name_ls = ["conv2d_8","conv2d_9","conv2d_10","conv2d_11","conv2d_12"]
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    private_pixels,y_val = load_data(sys.argv[1])
    private_pixels = [ private_pixels[i].reshape((1, 48, 48, 1)) 
                        for i in range(len(private_pixels)) ]
    choose_id = 17
    photo = private_pixels[choose_id]
    for cnt, fn in enumerate(collect_layers):
        print (fn)
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        '''
        img_path = os.path.join(vis_dir, store_path)
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))
        '''
        fig.savefig((name_ls[cnt])+'_layer.png')
main()
