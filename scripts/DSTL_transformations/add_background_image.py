#!/usr/bin/python                                                                                 
# -*- coding: iso-8859-1 -*-                                                                      

'''
    Implementation of random background adding to a specific image

    Author: Guillaume Sicard
'''

import sys, os, random
import cPickle
import Image, numpy           

class AddBackground():
    def __init__(self, threshold = 128, complexity = 1):
        self.h = 32
        self.w = 32
        self.threshold = 1;
        try: #in order to load locally if it is available
            self.bg_image_file = '/Tmp/image_net/'
            f=open(self.bg_image_file+'filelist.pkl')
        except:
            self.bg_image_file = '/data/lisa/data/ift6266h10/image_net/'
            f=open(self.bg_image_file+'filelist.pkl')
        self.image_files = cPickle.load(f)
        f.close()
        self.regenerate_parameters(complexity)
    
    def get_current_parameters(self):
        return [self.contrast]
    # get threshold value
    def get_settings_names(self):
        return ['contrast']
    
    # no need, except for testmod.py
    def regenerate_parameters(self, complexity):
        self.contrast = 1-numpy.random.rand()*complexity
        return [self.contrast]

    # load an image
    def load_image(self,filename):
        image = Image.open(filename).convert('L')
        image = numpy.asarray(image)
        image = (image / 255.0).astype(numpy.float32)
        return image

    # save an image
    def save_image(self,array, filename):
        image = (array * 255.0).astype('int')
        image = Image.fromarray(image)
        if (filename != ''):
            image.save(filename)
        else:
            image.show()

    # make a random 32x32 crop of an image
    def rand_crop(self,image):
        i_w, i_h = image.shape
        x, y = random.randint(0, i_w - self.w), random.randint(0, i_h - self.h)
        return image[x:x + self.w, y:y + self.h]

    # select a random background image from "bg_image_file" and crops it
    def rand_bg_image(self,maximage):
        i = random.randint(0, len(self.image_files) - 1)

        image = self.load_image(self.bg_image_file + self.image_files[i])
        self.bg_image = self.rand_crop(image)
        maxbg = self.bg_image.max()
        self.bg_image = self.bg_image / maxbg * ( max(maximage - self.contrast,0.0) ) 

    # set "bg_image" as background to "image", based on a pixels threshold
    def set_bg(self,image):
        tensor = numpy.asarray([self.bg_image,image],dtype='float32')
        return tensor.max(0)

    # transform an image file and return an array
    def transform_image_from_file(self, filename):
        self.rand_bg_image()
        image = self.load_image(filename)
        image = self.set_bg(image)
        return image

    # standard array to array transform
    def transform_image(self, image):
        self.rand_bg_image(image.max())
        image = self.set_bg(image)
        return image

    # test method
    def test(self,filename):
        import time

        sys.stdout.write('Starting addBackground test : loading image')
        sys.stdout.flush()

        image = self.load_image(filename)

        t = 0
        n = 500
        for i in range(n):
            t0 =  time.time()
            image2 = self.transform_image(image)
            t = ( i * t + (time.time() - t0) ) / (i + 1)
            sys.stdout.write('.')
            sys.stdout.flush()
            
        print "Done!\nAverage time : " + str(1000 * t) + " ms"

if __name__ == '__main__':

    myAddBackground = AddBackground()
    myAddBackground.test('./images/0-LiberationSans-Italic.ttf.jpg')
