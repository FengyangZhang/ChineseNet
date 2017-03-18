#!/usr/bin/python
# coding: utf-8

'''
Simple implementation of random contrast. This always switch half the time the polarity.
then it decides of a random contrast dependant of the complexity, the mean of the maximum and minimum
pixel value stays 0 (to avoid import bias change between exemples).

Author: Xavier Glorot
'''

import numpy as N
import copy


class Contrast():
    def __init__(self,complexity = 1):
        #---------- private attributes
        self.__nx__ = 32 #xdim of the images
        self.__ny__ = 32 #ydim of the images
        self.__Pinvert__ = 0.5 #probability to switch polarity
        self.__mincontrast__ = 0.15
        self.__resolution__ = 256
        self.__rangecontrastres__ = self.__resolution__ - N.int(self.__mincontrast__*self.__resolution__)
        #------------------------------------------------
        
        #---------- generation parameters
        self.regenerate_parameters(complexity)
        #------------------------------------------------
    
    def _get_current_parameters(self):
        return [self.invert,self.contrast]
    
    def get_settings_names(self):
        return ['invert','contrast']
    
    def regenerate_parameters(self, complexity):
        self.invert = (N.random.uniform() < self.__Pinvert__)
        self.contrast = self.__resolution__ - N.random.randint(1 + self.__rangecontrastres__ * complexity)
        return self._get_current_parameters()
    
    def transform_1_image(self,image): #the real transformation method
        maxi = image.max()
        mini = image.min()
        if self.invert:
            newimage = 1 - (self.__resolution__- self.contrast) / (2 * float(self.__resolution__)) -\
                        (image - mini) / float(maxi - mini) * self.contrast / float(self.__resolution__)
        else:
            newimage = (self.__resolution__- self.contrast) / (2 * float(self.__resolution__)) +\
                        (image - mini) / float(maxi - mini) * self.contrast / float(self.__resolution__)
        if image.dtype == 'uint8':
            return N.asarray(newimage*255,dtype='uint8')
        else:
            return N.asarray(newimage,dtype=image.dtype)
    
    def transform_image(self,image): #handling different format
        if image.shape == (self.__nx__,self.__ny__):
            return self.transform_1_image(image)
        if image.ndim == 3:
            newimage = copy.copy(image)
            for i in range(image.shape[0]):
                newimage[i,:,:] = self.transform_1_image(image[i,:,:])
            return newimage
        if image.ndim == 2 and image.shape != (self.__nx__,self.__ny__):
            newimage = N.reshape(image,(image.shape[0],self.__nx__,self.__ny__))
            for i in range(image.shape[0]):
                newimage[i,:,:] = self.transform_1_image(newimage[i,:,:])
            return N.reshape(newimage,image.shape)
        if image.ndim == 1:
            newimage = N.reshape(image,(self.__nx__,self.__ny__))
            newimage = self.transform_1_image(newimage)
            return N.reshape(newimage,image.shape)
        assert False #should never go there




#test on NIST (you need pylearn and access to NIST to do that)

if __name__ == '__main__':
    
    from pylearn.io import filetensor as ft
    import copy
    import pygame
    import time
    datapath = '/data/lisa/data/nist/by_class/'
    f = open(datapath+'digits/digits_train_data.ft')
    d = ft.read(f)
    
    pygame.surfarray.use_arraytype('numpy')
    
    pygame.display.init()
    screen = pygame.display.set_mode((8*2*32,8*32),0,8)
    anglcolorpalette=[(x,x,x) for x in xrange(0,256)]
    screen.set_palette(anglcolorpalette)
    
    MyContrast = Contrast()
    
    debut=time.time()
    MyContrast.transform_image(d)
    fin=time.time()
    print '------------------------------------------------'
    print d.shape[0],' images transformed in :', fin-debut, ' seconds'
    print '------------------------------------------------'
    print (fin-debut)/d.shape[0]*1000000,' microseconds per image'
    print '------------------------------------------------'
    print MyContrast.get_settings_names()
    print MyContrast._get_current_parameters()
    print MyContrast.regenerate_parameters(0)
    print MyContrast.regenerate_parameters(0.5)
    print MyContrast.regenerate_parameters(1)
    for i in range(10000):
        a=d[i,:]
        b=N.asarray(N.reshape(a,(32,32))).T
        
        new=pygame.surfarray.make_surface(b)
        new=pygame.transform.scale2x(new)
        new=pygame.transform.scale2x(new)
        new=pygame.transform.scale2x(new)
        new.set_palette(anglcolorpalette)
        screen.blit(new,(0,0))
        
        print MyContrast.get_settings_names(), MyContrast.regenerate_parameters(1)
        c=MyContrast.transform_image(a)
        b=N.asarray(N.reshape(c,(32,32))).T
        
        new=pygame.surfarray.make_surface(b)
        new=pygame.transform.scale2x(new)
        new=pygame.transform.scale2x(new)
        new=pygame.transform.scale2x(new)
        new.set_palette(anglcolorpalette)
        screen.blit(new,(8*32,0))
        
        pygame.display.update()
        raw_input('Press Enter')
    
    pygame.display.quit()
