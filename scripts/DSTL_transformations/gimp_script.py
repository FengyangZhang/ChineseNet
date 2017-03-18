#!/usr/bin/env python
# coding: utf-8

'''
Filtres GIMP sous Python
Auteur: Nicolas Boulanger-Lewandowski
Date: Hiver 2010

run with: gimp -i --batch-interpreter python-fu-eval --batch - < gimp_script.py
end with: pdb.gimp_quit(0)

Implémente le motionblur et le pinch
'''

from gimpfu import *
import numpy

img = gimp.Image(32, 32, GRAY)
img.disable_undo()
layer1 = gimp.Layer(img, "layer1", 32, 32, GRAY_IMAGE, 100, NORMAL_MODE)
img.add_layer(layer1, 0)
dest_rgn = layer1.get_pixel_rgn(0, 0, 32, 32, True)

def setpix(image):
    dest_rgn[:,:] = (image.T*255).astype(numpy.uint8).tostring()
    layer1.flush()
    layer1.update(0, 0, 32, 32)

def getpix():
    return numpy.fromstring(dest_rgn[:,:], 'UInt8').astype(numpy.float32).reshape((32,32)).T / 255.0

class GIMP1():
    def __init__(self, blur_bool = True):
        #This is used to avoid blurring for PNIST
        self.blur_bool = blur_bool

    def get_settings_names(self, blur_bool = True):
        return ['mblur_length', 'mblur_angle', 'pinch']
    
    def regenerate_parameters(self, complexity):
        if complexity:
           self.mblur_length = abs(int(round(numpy.random.normal(0, 3*complexity))))
        else:
            self.mblur_length = 0
        self.mblur_angle =  int(round(numpy.random.uniform(0,360)))
        self.pinch = numpy.random.uniform(-complexity, 0.7*complexity)

        return [self.mblur_length, self.mblur_angle, self.pinch]

    def transform_image(self, image):
        if self.mblur_length or self.pinch:
            setpix(image)
            if self.mblur_length and self.blur_bool:
                pdb.plug_in_mblur(img, layer1, 0, self.mblur_length, self.mblur_angle, 0, 0)
            if self.pinch:
                pdb.plug_in_whirl_pinch(img, layer1, 0.0, self.pinch, 1.0)
            image = getpix()

        return image

# test
if __name__ == '__main__':
    import Image
    im = numpy.asarray(Image.open("a.bmp").convert("L")) / 255.0

    test = GIMP1()
    print test.get_settings_names(), '=', test.regenerate_parameters(1)
    #for i in range(1000):
    im = test.transform_image(im)

    import pylab
    pylab.imshow(im, pylab.matplotlib.cm.Greys_r)
    pylab.show()

    pdb.gimp_quit(0)
