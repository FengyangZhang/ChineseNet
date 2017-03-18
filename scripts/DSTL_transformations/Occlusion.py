#!/usr/bin/python
# coding: utf-8

'''
Ajout de bruit d'occlusion dans l'image originale.

Le bruit provient d'un echantillon pris dans la seconde image puis rajoutee a
gauche ou a droite de l'image originale. De plus, il se peut aussi que le
bruit soit rajoute sur l'image originale, mais en plus pâle.

Le fichier /data/lisa/data/ift6266h10/echantillon_occlusion.ft 
(sur le reseau DIRO) est necessaire.

Il y a 30% de chance d'avoir une occlusion quelconque.

Sylvain Pannetier Lebeuf dans le cadre de IFT6266, hiver 2010

'''


import numpy

from pylearn.io import filetensor as ft

class Occlusion():
    
    def __init__(self,seed=9854):
        #Ces 4 variables representent la taille du "crop" sur l'image2
        #Ce "crop" est pris a partie de image1[15,15], le milieu de l'image1
        self.haut=2
        self.bas=2
        self.gauche=2
        self.droite=2
        
        #Ces deux variables representent le deplacement en x et y par rapport
        #au milieu du bord gauche ou droit
        self.x_arrivee=0
        self.y_arrivee=0
        
        #Cette variable =1 si l'image est mise a gauche et -1 si a droite
        #et =0 si au centre, mais plus pale
        self.endroit=-1
        
        #Cette variable determine l'opacite de l'ajout dans le cas ou on est au milieu
        self.opacite=0.5    #C'est completement arbitraire. Possible de le changer si voulu
        
        #Sert a dire si on fait quelque chose. 0=faire rien, 1 on fait quelque chose
        self.appliquer=1
        
        self.seed=seed
        #numpy.random.seed(self.seed)
        
        f3 = open('/data/lisa/data/ift6266h10/echantillon_occlusion.ft')   #Doit etre sur le reseau DIRO.
        #f3 = open('/home/sylvain/Dropbox/Msc/IFT6266/donnees/echantillon_occlusion.ft')
        #Il faut arranger le path sinon
        w=ft.read(f3)
        f3.close()
        
        self.longueur=len(w)
        self.d=(w.astype('float'))/255
        
        
    def get_settings_names(self):
        return ['haut','bas','gauche','droite','x_arrivee','y_arrivee','endroit','rajout','appliquer']
    
    def get_seed(self):
        return self.seed

    def regenerate_parameters(self, complexity):
        self.haut=min(15,int(numpy.abs(numpy.random.normal(int(8*complexity),2))))
        self.bas=min(15,int(numpy.abs(numpy.random.normal(int(8*complexity),2))))
        self.gauche=min(15,int(numpy.abs(numpy.random.normal(int(8*complexity),2))))
        self.droite=min(15,int(numpy.abs(numpy.random.normal(int(8*complexity),2))))
        if self.haut+self.bas+self.gauche+self.droite==0:   #Tres improbable
            self.haut=1
            self.bas=1
            self.gauche=1
            self.droite=1
        
        #Ces deux valeurs seront controlees afin d'etre certain de ne pas depasser
        self.x_arrivee=int(numpy.abs(numpy.random.normal(0,2))) #Complexity n'entre pas en jeu, pas besoin
        self.y_arrivee=int(numpy.random.normal(0,3)) 
        
        self.rajout=numpy.random.randint(0,self.longueur-1)  #les bouts de quelle lettre
        self.appliquer=numpy.random.binomial(1,0.4)    #####  40 % du temps, on met une occlusion #####
        
        if complexity == 0: #On ne fait rien dans ce cas
            self.applique=0
        
        self.endroit=numpy.random.randint(-1,2) 

        return self._get_current_parameters()

    def _get_current_parameters(self):
        return [self.haut,self.bas,self.gauche,self.droite,self.x_arrivee,self.y_arrivee,self.endroit,self.rajout,self.appliquer]
    
    
    def transform_image(self, image):
        if self.appliquer == 0: #Si on fait rien, on retourne tout de suite l'image
            return image
        
        #Attrapper le bruit d'occlusion
        bruit=self.d[self.rajout].reshape((32,32))[15-self.haut:15+self.bas+1,15-self.gauche:15+self.droite+1]
        
        if self.x_arrivee+self.gauche+self.droite>32:
            self.endroit*=-1    #On change de bord et on colle sur le cote
            self.x_arrivee=0
        if self.y_arrivee-self.haut <-16:
            self.y_arrivee=self.haut-16#On colle le morceau en haut
        if self.y_arrivee+self.bas > 15:
            self.y_arrivee=15-self.bas  #On colle le morceau en bas
            
        if self.endroit==-1:    #a gauche
            for i in xrange(-self.haut,self.bas+1):
                for j in xrange(0,self.gauche+self.droite+1):
                    image[16+self.y_arrivee+i,self.x_arrivee+j]=\
                    max(image[16+self.y_arrivee+i,self.x_arrivee+j],bruit[i+self.haut,j])
            
        elif self.endroit==1: #a droite
            for i in xrange(-self.haut,self.bas+1):
                for j in xrange(-self.gauche-self.droite,1):
                    image[16+self.y_arrivee+i,31-self.x_arrivee+j]=\
                    max(image[16+self.y_arrivee+i,31-self.x_arrivee+j],bruit[i+self.haut,j+self.gauche+self.droite])
            
        elif self.endroit==0:    #au milieu
            for i in xrange(-self.haut,self.bas+1):
                for j in xrange(-self.gauche,self.droite+1):
                    image[16+i,16+j]=max(image[16+i,16+j],bruit[i+self.haut,j+self.gauche]*self.opacite)
            
        
        return image
        
#---TESTS---

def _load_image():
    f = open('/home/sylvain/Dropbox/Msc/IFT6266/donnees/lower_test_data.ft')  #Le jeu de donnees est en local. 
    d = ft.read(f)
    w=numpy.asarray(d[numpy.random.randint(0,50)])
    return (w/255.0).astype('float')

def _test(complexite):
    
    transfo = Occlusion()
    for i in xrange(0,20):
        img = _load_image()
        pylab.imshow(img.reshape((32,32)))
        pylab.show()
        print transfo.get_settings_names()
        print transfo.regenerate_parameters(complexite)
        
        img_trans=transfo.transform_image(img.reshape((32,32)))
        
        print transfo.get_seed()
        pylab.imshow(img_trans.reshape((32,32)))
        pylab.show()
    

if __name__ == '__main__':
    import pylab
    import scipy
    _test(0.5)
