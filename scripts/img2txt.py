import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc 
import math
import re
import tables

def main(argv):
    img_dir = "../data/chars_generated/"
    img_srcs = os.listdir(img_dir)
    sys.stdout.flush()
    print('%d images in total.'%len(img_srcs))
    # leave files that are not jpg files
    is_jpg = re.compile(r'.+?\.jpg')

    print('generating training data...')
    counter = 0
    if (len(img_srcs)>0):
        hdf5_path = 'trainData.hdf5'
        #img_src = img_dir + img_srcs[0]
        #img_mat = np.array(Image.open(img_src))
        #img_mat = img2directMap(img_mat)

        hdf5_file = tables.open_file(hdf5_path, mode='w')
        filters = tables.Filters(complevel=5, complib='blosc')
        trainData = hdf5_file.create_earray(hdf5_file.root, 'trainData',
                                           tables.Atom.from_dtype(np.dtype('Int8')), 
                                           shape=(0, 18432),
                                           filters=filters,
                                           expectedrows=919975)

        for img_src in img_srcs:
            if(is_jpg.match(img_src)):
                counter = counter + 1
                if(counter%5000 == 0):
                    print("processing the %dth picture..." %counter)
                # if(counter > 500):
                #    break
                img_src = img_dir + img_src
                img_mat = np.array(Image.open(img_src))
                img_mat = img2directMap(img_mat)
                trainData.append(img_mat.reshape((1,18432)))
        hdf5_file.close()

def img2directMap(img):
    w_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    w_x = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    b = 0
    img_g_abs, img_g_arc = sobel(img, w_x, w_y, b)
    directMap = grad2directMap(img_g_abs, img_g_arc)
    return directMap

# sobel operation using conv            
def sobel(x, w_x, w_y, b):
    g_x = conv_forward_naive(x, w_x, b, {'stride': 1, 'pad': 1})
    g_y = conv_forward_naive(x, w_y, b, {'stride': 1, 'pad': 1})
    g_abs = np.abs(g_x) + np.abs(g_y) 
    # use arctan2 to compute [-pi, pi] angles
    g_arc = np.arctan2(g_y, g_x)
    return (g_abs, g_arc)

# map the gradients into directMap
def grad2directMap(g_abs, g_arc):
    pi = math.pi
    directMap = np.zeros((8, g_abs.shape[0], g_abs.shape[1]))
    arcs = np.array([0, pi/4, pi/2, pi*3/4, pi])
    for i in range(4):
        lbound = arcs[i]
        ubound = arcs[i+1]
        mask = (g_arc < ubound) * (g_arc >= lbound) 
        directMap[i, :, :] += mask * g_abs * (np.cos(g_arc - lbound) - np.sin(g_arc - lbound))
        directMap[i+1, :, :] += mask * g_abs * (np.cos(ubound - g_arc) - np.sin(ubound - g_arc))
    arcs = np.array([-pi, -pi*3/4, -pi/2, -pi/4, 0])
    for i in range(4):
        lbound = arcs[i]
        ubound = arcs[i+1]
        mask = (g_arc < ubound) * (g_arc >= lbound) 
        directMap[i+4, :, :] += mask * g_abs * (np.cos(g_arc - lbound) - np.sin(g_arc - lbound))
        if(i == 3):
            directMap[0, :, :] += mask * g_abs * (np.cos(ubound - g_arc) - np.sin(ubound - g_arc))
        else:
            directMap[i+5, :, :] += mask * g_abs * (np.cos(ubound - g_arc) - np.sin(ubound - g_arc))
    directMap[directMap>255] = 255
    return directMap

# a naive conv forward    
def conv_forward_naive(x, w, b, conv_param):
    out = None
    H, W = x.shape
    HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']

    x_padded = np.pad(x, ((pad, pad), (pad, pad)), 'constant')
    
    Out_h = (int)(1 + (H + 2 * pad - HH) / stride)
    Out_w = (int)(1 + (W + 2 * pad - WW) / stride)

    out = np.zeros((Out_h, Out_w))

    for k in range(Out_h):
        for l in range(Out_w):
            out[k, l] = np.sum(
                x_padded[k * stride : k * stride + HH, l * stride : l * stride + WW] * w) + b
            
    return out

if __name__ == '__main__':
    main(sys.argv[1:])
