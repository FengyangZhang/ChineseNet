#!/usr/bin/python
# coding: utf-8

'''
Ajout de bruit gaussien dans les donnees. A chaque iteration, un bruit poivre 
et sel est ajoute, puis un lissage gaussien autour de ce point est ajoute.
On fait un nombre d'iteration = 1024*complexity/25 ce qui equivaud
a complexity/25 des points qui recoivent le centre du noyau gaussien.
Il y en a beaucoup moins que le bruit poivre et sel, car la transformation
est plutôt aggressive et touche beaucoup de pixels autour du centre 

La grandeur de la gaussienne ainsi que son ecart type sont definit par complexity 
et par une composante aleatoire normale.

On a 25 % de chances d'effectuer le bruitage

Ce fichier prend pour acquis que les images sont donnees une a la fois
sous forme de numpy.array de 1024 (32 x 32) valeurs entre 0 et 1.

Sylvain Pannetier Lebeuf dans le cadre de IFT6266, hiver 2010

'''

import numpy
#import random
import scipy
from scipy import ndimage

class BruitGauss():
    
    def __init__(self,complexity=1,seed=6378):
        self.nb_chngmax =10 #Le nombre de pixels changes. Seulement pour fin de calcul
        self.grandeurmax = 20
        self.sigmamax = 6.0
        self.regenerate_parameters(complexity)
        self.seed=seed
        
        #numpy.random.seed(self.seed)
        
    def get_seed(self):
        return self.seed
        
    def get_settings_names(self):
        return ['nb_chng','sigma_gauss','grandeur']

    def regenerate_parameters(self, complexity):
        self.effectuer =numpy.random.binomial(1,0.25)    ##### On a 25% de faire un bruit #####

        
        if self.effectuer and complexity > 0:
            self.nb_chng=3+int(numpy.random.rand()*self.nb_chngmax*complexity)
            self.sigma_gauss=2.0 + numpy.random.rand()*self.sigmamax*complexity
            self.grandeur=12+int(numpy.random.rand()*self.grandeurmax*complexity)
            #creation du noyau gaussien
            self.gauss=numpy.zeros((self.grandeur,self.grandeur))
            x0 = y0 = self.grandeur/2.0
            for i in xrange(self.grandeur):
                for j in xrange(self.grandeur):
                    self.gauss[i,j]=numpy.exp(-((i-x0)**2 + (j-y0)**2) / self.sigma_gauss**2)
            #creation de la fenetre de moyennage
            self.moy=numpy.zeros((self.grandeur,self.grandeur))
            x0 = y0 = self.grandeur/2
            for i in xrange(0,self.grandeur):
                for j in xrange(0,self.grandeur):
                    self.moy[i,j]=((numpy.sqrt(2*(self.grandeur/2.0)**2) -\
                                 numpy.sqrt(numpy.abs(i-self.grandeur/2.0)**2+numpy.abs(j-self.grandeur/2.0)**2))/numpy.sqrt((self.grandeur/2.0)**2))**5
        else:
            self.sigma_gauss = 1 # eviter division par 0
            self.grandeur=1
            self.nb_chng = 0
            self.effectuer = 0
        
        return self._get_current_parameters()

    def _get_current_parameters(self):
        return [self.nb_chng,self.sigma_gauss,self.grandeur]

    
    def transform_image(self, image):
        if self.effectuer == 0:
            return image
        image=image.reshape((32,32))
        filtered_image = ndimage.convolve(image,self.gauss,mode='constant')
        assert image.shape == filtered_image.shape
        filtered_image = (filtered_image - filtered_image.min() + image.min()) / (filtered_image.max() - filtered_image.min() + image.min()) * image.max()
               
        #construction of the moyennage Mask
        Mask = numpy.zeros((32,32))
        
        for i in xrange(0,self.nb_chng):
            x_bruit=int(numpy.random.randint(0,32))
            y_bruit=int(numpy.random.randint(0,32))
            offsetxmin = 0
            offsetxmax = 0
            offsetymin = 0
            offsetymax = 0
            if x_bruit < self.grandeur / 2:
                offsetxmin = self.grandeur / 2 - x_bruit
            if 32-x_bruit < numpy.ceil(self.grandeur / 2.0):
                offsetxmax = numpy.ceil(self.grandeur / 2.0) - (32-x_bruit)
            if y_bruit < self.grandeur / 2:
                offsetymin = self.grandeur / 2 - y_bruit
            if 32-y_bruit < numpy.ceil(self.grandeur / 2.0):
                offsetymax = numpy.ceil(self.grandeur / 2.0) - (32-y_bruit)
            Mask[x_bruit - self.grandeur/2 + offsetxmin : x_bruit + numpy.ceil(self.grandeur/2.0) - offsetxmax,\
                    y_bruit - self.grandeur/2 + offsetymin : y_bruit + numpy.ceil(self.grandeur/2.0)-  offsetymax] +=\
                        self.moy[offsetxmin:self.grandeur - offsetxmax,offsetymin:self.grandeur - offsetymax] 
                    
        return numpy.asarray((image + filtered_image*Mask)/(Mask+1),dtype='float32')

#---TESTS---

def _load_image():
    f = open('/home/sylvain/Dropbox/Msc/IFT6266/donnees/lower_test_data.ft')  #Le jeu de donnees est en local. 
    d = ft.read(f)
    w=numpy.asarray(d[0])
    return (w/255.0).astype('float')

def _test(complexite):
    img=_load_image()
    transfo = BruitGauss()
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
    _test(0.5)


