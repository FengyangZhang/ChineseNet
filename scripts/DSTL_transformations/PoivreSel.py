#!/usr/bin/python
# coding: utf-8

'''
Ajout de bruit poivre et sel dans les donnees. Le bruit est distribue de facon 
aleatoire tire d'une uniforme tout comme la clarte des bites changees.

La proportion de bites aleatoires est definit par complexity/5.
Lorsque cette valeur est a 1 ==> Plus reconnaissable et 0 ==> Rien ne se passe

On a maintenant 25% de chance d'effectuer un bruitage.

Ce fichier prend pour acquis que les images sont donnees une a la fois
sous forme de numpy.array de 1024 (32 x 32) valeurs entre 0 et 1.

Sylvain Pannetier Lebeuf dans le cadre de IFT6266, hiver 2010

'''

import numpy
import random

class PoivreSel():
    
    def __init__(self,seed=9361):
        self.proportion_bruit=0.08 #Le pourcentage des pixels qui seront bruites
        self.nb_chng=10 #Le nombre de pixels changes. Seulement pour fin de calcul
        self.effectuer=1    #Vaut 1 si on effectue et 0 sinon.
        
        self.seed=seed
        #Les deux generateurs sont de types differents, avoir la meme seed n'a pas d'influence
        #numpy.random.seed(self.seed)
        #random.seed(self.seed)
        
    def get_seed(self):
        return self.seed
        
    def get_settings_names(self):
        return ['effectuer']
    
    def get_settings_names_determined_by_complexity(self,complexity):
        return ['proportion_bruit']

    def regenerate_parameters(self, complexity):
        self.proportion_bruit = float(complexity)/5
        self.nb_chng=int(1024*self.proportion_bruit)
        self.changements=random.sample(xrange(1024),self.nb_chng)   #Les pixels qui seront changes
        self.effectuer =numpy.random.binomial(1,0.25)    ##### On a 25% de faire un bruit #####
        return self._get_current_parameters()

    def _get_current_parameters(self):
        return [self.effectuer]
    
    def get_parameters_determined_by_complexity(self, complexity):
        return [float(complexity)/5]
    
    def transform_image(self, image):
        if self.effectuer == 0:
            return image
        
        image=image.reshape(1024,1)
        for j in xrange(0,self.nb_chng):
            image[self.changements[j]]=numpy.random.random()    #On determine les nouvelles valeurs des pixels changes
        return image.reshape(32,32)


#---TESTS---

def _load_image():
    f = open('/home/sylvain/Dropbox/Msc/IFT6266/donnees/lower_test_data.ft')  #Le jeu de donnees est en local. 
    d = ft.read(f)
    w=numpy.asarray(d[0])
    return (w/255.0).astype('float')

def _test(complexite):
    img=_load_image()
    transfo = PoivreSel()
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


