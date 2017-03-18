#!/usr/bin/python
# coding: utf-8

'''
Simple implementation of random affine transformations based on the Python 
Imaging Module affine transformations.


Author: Razvan Pascanu
'''

import numpy, Image



class AffineTransformation():
    def __init__( self, complexity = .5):
        self.shape = (32,32)
        self.complexity = complexity
        params = numpy.random.uniform(size=6) -.5
        self.a = 1. + params[0]*.6*complexity
        self.b = 0. + params[1]*.6*complexity
        self.c = params[2]*8.*complexity
        self.d = 0. + params[3]*.6*complexity
        self.e = 1. + params[4]*.6*complexity
        self.f = params[5]*8.*complexity

    
    def _get_current_parameters(self):
        return [self.a, self.b, self.c, self.d, self.e, self.f]

    def get_settings_names(self):
        return ['a','b','c','d','e','f']

    def regenerate_parameters(self, complexity):
        # generate random affine transformation
        # a point (x',y') of the new image corresponds to (x,y) of the old
        # image where : 
        #   x' = params[0]*x + params[1]*y + params[2]
        #   y' = params[3]*x + params[4]*y _ params[5]

        # the ranges are set manually as to look acceptable
 
        self.complexity = complexity
        params = numpy.random.uniform(size=6) -.5
        self.a = 1. + params[0]*.8*complexity
        self.b = 0. + params[1]*.8*complexity
        self.c = params[2]*9.*complexity
        self.d = 0. + params[3]*.8*complexity
        self.e = 1. + params[4]*.8*complexity
        self.f = params[5]*9.*complexity
        return self._get_current_parameters()

      


    def transform_image(self,NIST_image):
    
        im = Image.fromarray( \
                numpy.asarray(\
                       NIST_image.reshape(self.shape)*255.0, dtype='uint8'))
        nwim = im.transform( (32,32), Image.AFFINE, [self.a,self.b,self.c,self.d,self.e,self.f])
        return numpy.asarray(numpy.asarray(nwim)/255.0,dtype='float32')



if __name__ =='__main__':
    print 'random test'
    
    from pylearn.io import filetensor as ft
    import pylab

    datapath = '/data/lisa/data/nist/by_class/'

    f = open(datapath+'digits/digits_train_data.ft')
    d = ft.read(f)
    f.close()


    transformer = AffineTransformation()
    id = numpy.random.randint(30)
    
    pylab.figure()
    pylab.imshow(d[id].reshape((32,32)))
    pylab.figure()
    pylab.imshow(transformer.transform_image(d[id]).reshape((32,32)))

    pylab.show()

