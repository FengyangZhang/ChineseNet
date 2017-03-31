import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc 
import math
import re
from CCBReader import CCBReader
import scipy.ndimage
import math
def main(argv):
    reader = CCBReader("./000.ccb")
    reader.printBasicInfo()

    data_path = 'trainData.hdf5'
    data_file = tables.open_file(data_path, mode='a')
    filters = tables.Filters(complevel=5, complib='blosc')
    trainData = data_file.create_earray(data_file.root, 'trainData',
                                       tables.Atom.from_dtype(np.dtype('uint8')), 
                                       shape=(0, 8192),
                                       filters=filters)
    
    label_path = 'trainLabel.hdf5'
    label_file = tables.open_file(label_path, mode='a')
    filters = tables.Filters(complevel=5, complib='blosc')
    trainLabel = label_file.create_earray(label_file.root, 'trainLabel',
                                       tables.Atom.from_dtype(np.dtype('uint32')), 
                                       shape=(0,1),
                                       filters=filters)

    for i in range(reader.num_chars):
        if(i % 1000 == 0):
            print("processing %dth image..." %i)
        img_array = reader.lookUpImgByOffset(i)
        img_array = invert(img_array)
        directMap = img2directMap(img_array)
        trainData.append(img_mat.reshape((1,8192)))
        trainLabel.append(np.array(i).reshape(1,1))
    data_file.close()
    label_file.close()

    # img_array = reader.lookUpImgByOffset(0)
    # img_array = invert(img_array)
    # rotation(img_array)
    # plt.imshow(img_array, cmap='gray')
    # plt.gca().axis('off')
    # plt.show()

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

def interpolation(img_array):
    return scipy.misc.imresize(img_array, (32, 32), interp='bilinear')

# invert the image
def invert(img_array):
    return np.invert(img_array)

# shifting pixels
def shift(img_array):
    return scipy.ndimage.interpolation.affine_transform()

# gaussian
def gussianNoise():
    print('performing gaussian blur on the original images...')
    for name in img_names:
          image = Image.open(self.in_dir + name)
          image = image.filter(ImageFilter.GaussianBlur(radius=1)) 
          image.save(self.out_dir + name.strip('g.jpg') + 'g.jpg')
    print('gaussian blur completed.')

# rotation
def rotation(img_array):
    c_in=0.5*np.array(img_array.shape)
    c_out=np.array((32,32))
    for i in xrange(-3, 4):
        a=i*15.0*math.pi/180.0
        transform=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
        offset=c_in-c_out.dot(transform)
        dst=scipy.ndimage.interpolation.affine_transform(
            img_array,transform.T,order=2,offset=offset,output_shape=(32,32),cval=0.0)
        plt.subplot(1,7,i+4);plt.axis('off');plt.imshow(dst, cmap='gray')
    plt.show()
#threshold
def threshold():
    for name in img_names:
        data = np.array(Image.open(self.in_dir+name))
        index = data >=  128
        data[index] = 255
        index = data < 128
        data[index] = 0
        image = Image.fromarray(data)
        image.save(self.out_dir + name)

if __name__ == '__main__':
    main(sys.argv[1:])
