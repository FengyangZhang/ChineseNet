#!/usr/bin/python
# coding: utf-8

'''
Simple implementation of random thickness deformation using morphological
operation of scipy.
Only one morphological operation applied (dilation or erosion), the kernel is random
out of a list of 12 symmetric kernels. (only 5 to be chosen for erosion because it can
hurt the recognizability of the charater and 12 for dilation).

Author: Xavier Glorot

'''

import scipy.ndimage.morphology
import numpy as N


class Thick():
    def __init__(self,complexity = 1):
        #---------- private attributes
        self.__nx__ = 32 #xdim of the images
        self.__ny__ = 32 #ydim of the images
        self.__erodemax__ = 5 #nb of index max of erode structuring elements
        self.__dilatemax__ = 9 #nb of index max of dilation structuring elements
        self.__structuring_elements__ = [N.asarray([[1,1]]),N.asarray([[1],[1]]),\
                                        N.asarray([[1,1],[1,1]]),N.asarray([[0,1,0],[1,1,1],[0,1,0]]),\
                                        N.asarray([[1,1,1],[1,1,1]]),N.asarray([[1,1],[1,1],[1,1]]),\
                                        N.asarray([[1,1,1],[1,1,1],[1,1,1]]),\
                                        N.asarray([[1,1,1,1],[1,1,1,1],[1,1,1,1]]),\
                                        N.asarray([[1,1,1],[1,1,1],[1,1,1],[1,1,1]]),\
                                        N.asarray([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]]),\
                                        N.asarray([[1,1,1,1],[1,1,1,1]]),N.asarray([[1,1],[1,1],[1,1],[1,1]])]
        #------------------------------------------------
        
        #---------- generation parameters
        self.regenerate_parameters(complexity)
        #------------------------------------------------
    
    def _get_current_parameters(self):
        return [self.thick_param]
    
    def get_settings_names(self):
        return ['thick_param']
    
    def regenerate_parameters(self, complexity):
        self.erodenb = N.ceil(complexity * self.__erodemax__)
        self.dilatenb = N.ceil(complexity * self.__dilatemax__)
        self.Perode = self.erodenb / (self.dilatenb + self.erodenb + 1.0)
        self.Pdilate = self.dilatenb / (self.dilatenb   + self.erodenb + 1.0)
        assert (self.Perode + self.Pdilate <= 1) & (self.Perode + self.Pdilate >= 0)
        assert (complexity >= 0) & (complexity <= 1)
        P = N.random.uniform()
        if P>1-(self.Pdilate+self.Perode):
            if P>1-(self.Pdilate+self.Perode)+self.Perode:
                self.meth = 1
                self.nb=N.random.randint(self.dilatenb)
            else:
                self.meth = -1
                self.nb=N.random.randint(self.erodenb)
        else:
            self.meth = 0
            self.nb = -1
        self.thick_param = self.meth*self.nb
        return self._get_current_parameters()
    
    def transform_1_image(self,image): #the real transformation method
        if self.meth!=0:
            maxi = float(N.max(image))
            mini = float(N.min(image))
            
            imagenorm=image/maxi
            
            if self.meth==1:
                trans=scipy.ndimage.morphology.grey_dilation\
                    (imagenorm,size=self.__structuring_elements__[self.nb].shape,structure=self.__structuring_elements__[self.nb])
            else:
                trans=scipy.ndimage.morphology.grey_erosion\
                    (imagenorm,size=self.__structuring_elements__[self.nb].shape,structure=self.__structuring_elements__[self.nb])
            
            #------renormalizing
            maxit = N.max(trans)
            minit = N.min(trans)
            trans= N.asarray((trans - (minit+mini)) / (maxit - (minit+mini)) * maxi,dtype=image.dtype)
            #--------
            return trans
        else:
            return image
    
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
    screen = pygame.display.set_mode((8*4*32,8*32),0,8)
    anglcolorpalette=[(x,x,x) for x in xrange(0,256)]
    screen.set_palette(anglcolorpalette)
    
    MyThick = Thick()
    
    #debut=time.time()
    #MyThick.transform_image(d)
    #fin=time.time()
    #print '------------------------------------------------'
    #print d.shape[0],' images transformed in :', fin-debut, ' seconds'
    #print '------------------------------------------------'
    #print (fin-debut)/d.shape[0]*1000000,' microseconds per image'
    #print '------------------------------------------------'
    #print MyThick.get_settings_names()
    #print MyThick._get_current_parameters()
    #print MyThick.regenerate_parameters(0)
    #print MyThick.regenerate_parameters(0.5)
    #print MyThick.regenerate_parameters(1)
    for i in range(10000):
        a=d[i,:]
        b=N.asarray(N.reshape(a,(32,32))).T
        
        new=pygame.surfarray.make_surface(b)
        new=pygame.transform.scale2x(new)
        new=pygame.transform.scale2x(new)
        new=pygame.transform.scale2x(new)
        new.set_palette(anglcolorpalette)
        screen.blit(new,(0,0))
        
        #max dilation
        MyThick.meth=1
        MyThick.nb=MyThick.__dilatemax__
        c=MyThick.transform_image(a)
        b=N.asarray(N.reshape(c,(32,32))).T
        
        new=pygame.surfarray.make_surface(b)
        new=pygame.transform.scale2x(new)
        new=pygame.transform.scale2x(new)
        new=pygame.transform.scale2x(new)
        new.set_palette(anglcolorpalette)
        screen.blit(new,(8*32,0))
        
        #max erosion
        MyThick.meth=-1
        MyThick.nb=MyThick.__erodemax__
        c=MyThick.transform_image(a)
        b=N.asarray(N.reshape(c,(32,32))).T
        
        new=pygame.surfarray.make_surface(b)
        new=pygame.transform.scale2x(new)
        new=pygame.transform.scale2x(new)
        new=pygame.transform.scale2x(new)
        new.set_palette(anglcolorpalette)
        screen.blit(new,(8*2*32,0))
        
        #random
        print MyThick.get_settings_names(), MyThick.regenerate_parameters(1)
        c=MyThick.transform_image(a)
        b=N.asarray(N.reshape(c,(32,32))).T
        
        new=pygame.surfarray.make_surface(b)
        new=pygame.transform.scale2x(new)
        new=pygame.transform.scale2x(new)
        new=pygame.transform.scale2x(new)
        new.set_palette(anglcolorpalette)
        screen.blit(new,(8*3*32,0))
        
        pygame.display.update()
        raw_input('Press Enter')
    
    pygame.display.quit()
