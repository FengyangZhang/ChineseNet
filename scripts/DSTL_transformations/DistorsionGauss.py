#!/usr/bin/python
# coding: utf-8

'''
Ajout d'une composante aleatoire dans chaque pixel de l'image.
C'est une distorsion gaussienne de moyenne 0 et d'écart type complexity/10

Il y a 30% d'effectuer le bruitage

Sylvain Pannetier Lebeuf dans le cadre de IFT6266, hiver 2010

'''

import numpy
import random

class DistorsionGauss():
    
    def __init__(self,seed=3459):
        self.ecart_type=0.1 #L'ecart type de la gaussienne
        self.effectuer=1    #1=on effectue et 0=rien faire
        self.seed=seed
        
        #Les deux generateurs sont de types differents, avoir la meme seed n'a pas d'influence
        #numpy.random.seed(self.seed) 
        #random.seed(self.seed)
        
    def get_settings_names(self):
        return ['effectuer']
    
    def get_seed(self):
        return self.seed
    
    def get_settings_names_determined_by_complexity(self,complexity):
        return ['ecart_type']

    def regenerate_parameters(self, complexity):
        self.ecart_type=float(complexity)/10
        self.effectuer =numpy.random.binomial(1,0.3)    ##### On a 30% de faire un bruit #####
        return self._get_current_parameters()

    def _get_current_parameters(self):
        return [self.effectuer]
    
    def get_parameters_determined_by_complexity(self,complexity):
        return [float(complexity)/10]
    
    def transform_image(self, image):
        if self.effectuer == 0:
            return image
        
        image=image.reshape(1024,1)
        aleatoire=numpy.zeros((1024,1)).astype('float32')
        for i in xrange(0,1024):
            aleatoire[i]=float(random.gauss(0,self.ecart_type))
        image=image+aleatoire
        
        
        #Ramener tout entre 0 et 1. Ancienne facon de normaliser.
        #Resultats moins interessant je trouve.
##        if numpy.min(image) < 0:
##            image-=numpy.min(image)
##        if numpy.max(image) > 1:
##            image/=numpy.max(image)
            
        for i in xrange(0,1024):
            image[i]=min(1,max(0,image[i]))
            
        return image.reshape(32,32)


#---TESTS---

def _load_image():
    f = open('/home/sylvain/Dropbox/Msc/IFT6266/donnees/lower_test_data.ft')  #Le jeu de donnees est en local. 
    d = ft.read(f)
    w=numpy.asarray(d[random.randint(0,100)])
    return (w/255.0).astype('float')

def _test(complexite):
    img=_load_image()
    transfo = DistorsionGauss()
    pylab.imshow(img.reshape((32,32)))
    pylab.show()
    print transfo.get_settings_names()
    print transfo.regenerate_parameters(complexite)
    
    img_trans=transfo.transform_image(img)
    
    pylab.imshow(img_trans.reshape((32,32)))
    pylab.show()
    

if __name__ == '__main__':
    from pylearn.io import filetensor as ft
    import pylab
    for i in xrange(0,5):
        _test(0.5)


