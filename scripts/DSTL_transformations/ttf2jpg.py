#!/usr/bin/python                                                                                 
# -*- coding: iso-8859-1 -*-                                                                      

'''
    Implementation of font image generator
    download fonts from http://www.dafont.com for exemple

    Author: Guillaume Sicard
'''

import sys, os, fnmatch, random
import Image, ImageFont, ImageDraw, numpy
import cPickle

class ttf2jpg():
    def __init__(self, font_file = ''):
        self.w = 32
        self.h = 32
        self.font_dir = '/Tmp/allfonts/'
        self.font_file = font_file
        self.image_dir = './images/'
        self.pattern = '*.ttf'
        self.char_list = []
        for i in range(0,10):
            self.char_list.append(chr(ord('0') + i) )
        for i in range(0,26):
            self.char_list.append(chr(ord('A') + i) )
        for i in range(0,26):
            self.char_list.append(chr(ord('a') + i) )
        f = open( self.font_dir + 'filelist.pkl' ,'r')
        self.font_files = cPickle.load(f)
        f.close()

    # get font name
    def get_settings_names(self):
        return [self.font_file]

    # save an image
    def save_image(self,array, filename = ''):
        image = (array * 255.0).astype('int')
        image = Image.fromarray(image).convert('L')
        if (filename != ''):
            image.save(filename)
        else:
            image.show()

    # set a random font for character generation
    def set_random_font(self):
        i = random.randint(0, len(self.font_files) - 1)
        self.font_file = self.font_dir + self.font_files[i]

    # return a picture array of "text" with font "font_file"
    def create_image(self, text):
         # create a w x h black picture, and a drawing space
        image = Image.new('L', (self.w, self.h), 'Black')
        draw = ImageDraw.Draw(image)

        # load the font with the right size
        font = ImageFont.truetype(self.font_file, 28)
        d_w,d_h =  draw.textsize(text, font=font)

        # write text and aligns it
        draw.text(((32 - d_w) / 2, ((32 - d_h) / 2)), text, font=font, fill='White')

        image = numpy.asarray(image)
        image = (image / 255.0).astype(numpy.float32)

        return image

    # write all the letters and numbers into pictures
    def process_font(self):
        for i in range(0, len(self.char_list) ):
            image = self.create_image(self.char_list[i])
            self.save_image(image, self.image_dir + self.char_list[i] + '-' + os.path.basename(self.font_file) + '.jpg')
            sys.stdout.write('.')
            sys.stdout.flush()
        return (len(self.char_list))

    # generate the character from the font_file and returns a numpy array
    def generate_image_from_char(self, character, font_file = ''):
        if (font_file != ''):
            self.font_file = font_file

        return self.create_image(character)

    # generate random character from random font file as a numpy array
    def generate_image(self):
        self.set_random_font()
        i = random.randint(0, len(self.char_list) - 1)
        return self.generate_image_from_char(self.char_list[i]), i

    # test method, create character images for all fonts in "font_dir" in dir "image_dir"
    def test(self):
        import time

        # look for ttf files
        files = os.listdir(self.font_dir)
        font_files = fnmatch.filter(files, self.pattern)

        # create "image_dir" if it doesn't exist
        if not os.path.isdir(self.image_dir):
            os.mkdir(self.image_dir)

        sys.stdout.write( str(len(font_files)) + ' fonts found, generating jpg images in folder ' + self.image_dir )
        sys.stdout.flush()

        # main loop
        t =  time.time()
        n = 0

        for font_file in font_files:
            self.font_file = self.font_dir + font_file
            n += self.process_font()
        t = time.time() - t

        sys.stdout.write('\nall done!\n' + str(n) + ' images generated in ' + str(t) + 's (average : ' + str(1000 * t / n) + ' ms/im)\n')

if __name__ == '__main__':

    myttf2jpg = ttf2jpg()
    #myttf2jpg.test()
    image, i = myttf2jpg.generate_image()
    myttf2jpg.save_image(image, '')
