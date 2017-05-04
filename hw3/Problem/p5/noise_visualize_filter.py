import os, sys
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from keras.utils import np_utils
import numpy as np

NUM_STEPS = 10
RECORD_FREQ = 2


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_img_data,iter_func):
    filter_images = (input_img_data.reshape(48,48),0)
    for i in range(num_step):
        target, grads_value = iter_func([input_img_data,0])
        input_img_data += grads_value 
        if target <= 0.:
        # some filters get stuck to 0, we can skip them
            break

        # decode the resulting input image
        if target > 0:
            img = deprocess_image(input_img_data[0])
            filter_images = (img.reshape(48,48),target)
    return filter_images

def main():
    emotion_classifier = load_model(sys.argv[1])
    emotion_classifier.summary()
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    print (layer_dict)
    input_img = emotion_classifier.input

    name_ls = ["conv2d_10","conv2d_8","conv2d_12"]
    collect_layers = [ layer_dict[name].output for name in name_ls ]

    for cnt, c in enumerate(collect_layers):
        filter_imgs = []
        nb_filter = int(c.shape[3])
        for filter_idx in range(nb_filter):
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img,K.learning_phase()],[target, grads])
           
            a = grad_ascent(20, input_img_data, iterate)
            filter_imgs.append(a)
        

        fig = plt.figure(figsize=(14, 8))
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(filter_imgs[i][0], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel('{:.3f}'.format(filter_imgs[i][1]))
            plt.tight_layout()
        fig.suptitle('Filters of layer {} '.format(name_ls[cnt]))
        print ('saveing '+name_ls[cnt])
        fig.savefig(os.path.join('./','{}'.format(name_ls[cnt], RECORD_FREQ)))
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x
if __name__ == "__main__":
    main()
