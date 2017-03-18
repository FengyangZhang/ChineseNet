#!/usr/bin/python
# coding: utf-8

'''
Ajout d'une rature sur le caractère. La rature est en fait un 1 qui recoit une
rotation et qui est ensuite appliqué sur le caractère. Un grossissement, puis deux
erosions sont effectuees sur le 1 afin qu'il ne soit plus reconnaissable.
Il y a des chances d'avoir plus d'une seule rature !

Il y a 15% d'effectuer une rature.

Ce fichier prend pour acquis que les images sont donnees une a la fois
sous forme de numpy.array de 1024 (32 x 32) valeurs entre 0 et 1.

Sylvain Pannetier Lebeuf dans le cadre de IFT6266, hiver 2010

'''

import numpy, Image, random
import scipy.ndimage.morphology
from pylearn.io import filetensor as ft


class Rature():
   
    def __init__(self,seed=1256):
        self.angle=0 #Angle en degre de la rotation (entre 0 et 180)
        self.numero=0 #Le numero du 1 choisi dans la banque de 1
        self.gauche=-1   #Le numero de la colonne la plus a gauche contenant le 1
        self.droite=-1
        self.haut=-1
        self.bas=-1
        self.faire=1    #1=on effectue et 0=fait rien
        
        self.crop_haut=0
        self.crop_gauche=0  #Ces deux valeurs sont entre 0 et 31 afin de definir
                            #l'endroit ou sera pris le crop dans l'image du 1
                            
        self.largeur_bande=-1    #La largeur de la bande
        self.smooth=-1   #La largeur de la matrice carree servant a l'erosion
        self.nb_ratures=-1   #Le nombre de ratures appliques
        self.fini=0 #1=fini de mettre toutes les couches 0=pas fini
        self.complexity=0   #Pour garder en memoire la complexite si plusieurs couches sont necessaires
        self.seed=seed
        
        #numpy.random.seed(self.seed)
        
        f3 = open('/data/lisa/data/ift6266h10/un_rature.ft')   #Doit etre sur le reseau DIRO.
        #f3 = open('/home/sylvain/Dropbox/Msc/IFT6266/donnees/un_rature.ft')
        #Il faut arranger le path sinon
        w=ft.read(f3)
        f3.close()
        self.d=(w.astype('float'))/255
        
        self.patch=self.d[0].reshape((32,32)) #La patch de rature qui sera appliquee sur l'image

    def get_settings_names(self):
        return ['angle','numero','faire','crop_haut','crop_gauche','largeur_bande','smooth','nb_ratures']
    
    def get_seed(self):
        return self.seed

    def regenerate_parameters(self, complexity,next_rature = False):
        
        
        self.numero=random.randint(0,4999)  #Ces bornes sont inclusives !
        self.fini=0
        self.complexity=complexity
            
        if float(complexity) > 0:
            
            self.gauche=self.droite=self.haut=self.bas=-1   #Remet tout a -1
            
            self.angle=int(numpy.random.normal(90,100*complexity))

            self.faire=numpy.random.binomial(1,0.15)    ##### 15% d'effectuer une rature #####
            if next_rature:
                self.faire = 1
            #self.faire=1 #Pour tester seulement
            
            self.crop_haut=random.randint(0,17)
            self.crop_gauche=random.randint(0,17)
            if complexity <= 0.25 :
                self.smooth=6
            elif complexity <= 0.5:
                self.smooth=5
            elif complexity <= 0.75:
                self.smooth=4
            else:
                self.smooth=3
            
            p = numpy.random.rand()
            if p < 0.5:
                self.nb_ratures= 1
            else:
                if p < 0.8:
                    self.nb_ratures = 2
                else:
                    self.nb_ratures = 3
            
            #Creation de la "patch" de rature qui sera appliquee sur l'image
            if self.faire == 1:
                self.get_size()
                self.get_image_rot()    #On fait la "patch"
            
        else:
            self.faire=0    #On ne fait rien si complexity=0 !!
        
        return self._get_current_parameters()
    
    
    def get_image_rot(self):
        image2=(self.d[self.numero].reshape((32,32))[self.haut:self.bas,self.gauche:self.droite])
        
        im = Image.fromarray(numpy.asarray(image2*255,dtype='uint8'))
        
        #La rotation et le resize sont de belle qualite afin d'avoir une image nette
        im2 = im.rotate(self.angle,Image.BICUBIC,expand=False)
        im3=im2.resize((50,50),Image.ANTIALIAS)
        
        grosse=numpy.asarray(numpy.asarray(im3)/255.0,dtype='float32')
        crop=grosse[self.haut:self.haut+32,self.gauche:self.gauche+32]
        
        self.get_patch(crop)
        
    def get_patch(self,crop):
        smooting = numpy.ones((self.smooth,self.smooth))
        #Il y a deux erosions afin d'avoir un beau resultat. Pas trop large et
        #pas trop mince
        trans=scipy.ndimage.morphology.grey_erosion\
                    (crop,size=smooting.shape,structure=smooting,mode='wrap')
        trans1=scipy.ndimage.morphology.grey_erosion\
                    (trans,size=smooting.shape,structure=smooting,mode='wrap')
        
               
        patch_img=Image.fromarray(numpy.asarray(trans1*255,dtype='uint8'))
        
        patch_img2=patch_img.crop((4,4,28,28)).resize((32,32))  #Pour contrer les effets de bords !
        
        trans2=numpy.asarray(numpy.asarray(patch_img2)/255.0,dtype='float32')
            
            
        #Tout ramener entre 0 et 1
        trans2=trans2-trans2.min() #On remet tout positif
        trans2=trans2/trans2.max()
        
        #La rayure a plus de chance d'etre en bas ou oblique le haut a 10h
        if random.random() <= 0.5:  #On renverse la matrice dans ce cas
            for i in xrange(0,32):
                self.patch[i,:]=trans2[31-i,:]
        else:
            self.patch=trans2
        
    
    
    
    def get_size(self):
        image=self.d[self.numero].reshape((32,32))
        
        #haut
        for i in xrange(0,32):
            for j in xrange(0,32):
                if(image[i,j]) != 0:
                    if self.haut == -1:
                        self.haut=i
                        break
            if self.haut > -1:
                break
        
        #bas
        for i in xrange(31,-1,-1):
            for j in xrange(0,32):
                if(image[i,j]) != 0:
                    if self.bas == -1:
                        self.bas=i
                        break
            if self.bas > -1:
                break
            
        #gauche
        for i in xrange(0,32):
            for j in xrange(0,32):
                if(image[j,i]) != 0:
                    if self.gauche == -1:
                        self.gauche=i
                        break
            if self.gauche > -1:
                break
            
        #droite
        for i in xrange(31,-1,-1):
            for j in xrange(0,32):
                if(image[j,i]) != 0:
                    if self.droite == -1:
                        self.droite=i
                        break
            if self.droite > -1:
                break
                

    def _get_current_parameters(self):
        return [self.angle,self.numero,self.faire,self.crop_haut,self.crop_gauche,self.largeur_bande,self.smooth,self.nb_ratures]

    def transform_image(self, image):
        if self.faire == 0: #Rien faire !!
            return image
        
        if self.fini == 0:   #S'il faut rajouter des couches
            patch_temp=self.patch
            for w in xrange(1,self.nb_ratures):
                self.regenerate_parameters(self.complexity,1)
                for i in xrange(0,32):
                    for j in xrange(0,32):
                        patch_temp[i,j]=max(patch_temp[i,j],self.patch[i,j])
            self.fini=1
            self.patch=patch_temp
            
        for i in xrange(0,32):
            for j in xrange(0,32):
                image[i,j]=max(image[i,j],self.patch[i,j])
        self.patch*=0   #Remise a zero de la patch (pas necessaire)
        return image


#---TESTS---

def _load_image():
    f = open('/home/sylvain/Dropbox/Msc/IFT6266/donnees/lower_test_data.ft')  #Le jeu de donnees est en local. 
    d = ft.read(f)
    w=numpy.asarray(d[0:1000])
    return (w/255.0).astype('float')

def _test(complexite):
    img=_load_image()
    transfo = Rature()
    for i in xrange(0,10):
        img2=img[random.randint(0,1000)]
        pylab.imshow(img2.reshape((32,32)))
        pylab.show()
        print transfo.get_settings_names()
        print transfo.regenerate_parameters(complexite)
        img2=img2.reshape((32,32))
        
        img2_trans=transfo.transform_image(img2)
        
        pylab.imshow(img2_trans.reshape((32,32)))
        pylab.show()
    

if __name__ == '__main__':
    from pylearn.io import filetensor as ft
    import pylab
    _test(1)


