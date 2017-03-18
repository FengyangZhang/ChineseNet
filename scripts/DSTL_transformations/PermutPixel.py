#!/usr/bin/python
# coding: utf-8

'''
Un echange de pixels est effectue entre certain pixels choisit aleatoirement
et un de ses 4 voisins, tout aussi choisi aleatoirement.

Le nombre de pixels permutes est definit pas complexity*1024

Il y a proba 20% d'effectuer le bruitage

Sylvain Pannetier Lebeuf dans le cadre de IFT6266, hiver 2010

'''

import numpy
import random

class PermutPixel():
    
    def __init__(self,seed=7152):
        self.nombre=10 #Le nombre de pixels a permuter
        self.proportion=0.3
        self.effectuer=1    #1=on effectue, 0=rien faire
        self.seed=seed
        
        #Les deux generateurs sont de types differents, avoir la meme seed n'a pas d'influence
        #numpy.random.seed(self.seed)
        #random.seed(self.seed)
        
    def get_seed(self):
        return self.seed
        
    def get_settings_names(self):
        return ['effectuer']
    
    def get_settings_names_determined_by_complexity(self,complexity):
        return ['nombre']

    def regenerate_parameters(self, complexity):
        self.proportion=float(complexity)/3
        self.nombre=int(256*self.proportion)*4   #Par multiple de 4 (256=1024/4)
        self.echantillon=random.sample(xrange(0,1024),self.nombre)  #Les pixels qui seront permutes
        self.effectuer =numpy.random.binomial(1,0.2)    ##### On a 20% de faire un bruit #####
        return self._get_current_parameters()

    def _get_current_parameters(self):
        return [self.effectuer]  
    
    def get_parameters_determined_by_complexity(self, complexity):
        return [int(complexity*256)*4]
    
    def transform_image(self, image):
        if self.effectuer==0:
            return image
        
        image=image.reshape(1024,1)
        temp=0  #variable temporaire

        for i in xrange(0,self.nombre,4):   #Par bonds de 4
            #gauche
            if self.echantillon[i] > 0:
                temp=image[self.echantillon[i]-1]
                image[self.echantillon[i]-1]=image[self.echantillon[i]]
                image[self.echantillon[i]]=temp
            #droite
            if self.echantillon[i+1] < 1023:
                temp=image[self.echantillon[i+1]+1]
                image[self.echantillon[i+1]+1]=image[self.echantillon[i+1]]
                image[self.echantillon[i+1]]=temp
            #haut
            if self.echantillon[i+2] > 31:
                temp=image[self.echantillon[i+2]-32]
                image[self.echantillon[i+2]-32]=image[self.echantillon[i+2]]
                image[self.echantillon[i+2]]=temp
            #bas
            if self.echantillon[i+3] < 992:
                temp=image[self.echantillon[i+3]+32]
                image[self.echantillon[i+3]+32]=image[self.echantillon[i+3]]
                image[self.echantillon[i+3]]=temp
            
            
        return image.reshape((32,32))


#---TESTS---

def _load_image():
    f = open('/home/sylvain/Dropbox/Msc/IFT6266/donnees/lower_test_data.ft')  #Le jeu de donnees est en local. 
    d = ft.read(f)
    w=numpy.asarray(d[random.randint(0,100)])
    return (w/255.0).astype('float')

def _test(complexite):
    img=_load_image()
    transfo = PermutPixel()
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


