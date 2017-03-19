import sys
import getopt
from PIL import Image, ImageFilter, ImageChops
import numpy as np
import os
import re
import PIL.ImageOps
import cv2
import glob

class dataGenerator:
  in_dir = None
  out_dir = None
  img_names = []

  def __init__(self, in_dir, out_dir):
    self.in_dir = in_dir
    self.out_dir = out_dir
    self.img_names = glob.glob(in_dir + '*.jpg')      
    if(len(self.img_names) == 0):
      print("[WARNING] No picture in the directory!")
    else:
      for i in range(len(self.img_names)):
        self.img_names[i] = self.img_names[i].split('/')[-1]

  # do interpolation to generate (32,32) images
  def interpolation():
    for name in img_names:
      img = cv2.imread(self.in_dir + name)
      res = cv2.resize(img, (32, 32), interpolation = cv2.INTER_CUBIC)
      cv2.imwrite(self.out_dir + name, res)

  # resize the image
  def resize():
    for name in img_names:
      image = Image.open(self.in_dir + name)
      image = image.resize((48, 48), Image.ANTIALIAS)
      image.save(self.out_dir + name)

  # invert the image
  def invert():
    for name in img_names:
      image = Image.open(self.in_dir + name)
      inverted_image = PIL.ImageOps.invert(image)
      inverted_image.save(self.out_dir + name)

  # convert to grayscale
  def toGrayscale():
    for name in img_names:
      image = Image.open(self.in_dir + name).convert('L')
      image.save(self.out_dir + name)

  # shifting pixels
  def shift():
    for name in img_names:
      img_orig = Image.open(self.in_dir + name)
      for i in range(-3, 4):
        for j in range(-3, 4):
          img_gen = ImageChops.offset(img_orig, 3*i, 3*j)
          img_gen.save(self.out_dir + name.strip('.jpg') + '_%d%d.jpg' %(i, j))
    print('pixel shifting completed.')

  # gaussian
  def gussianNoise():
    print('performing gaussian blur on the original images...')
    for name in img_names:
      image = Image.open(self.in_dir + name)
      image = image.filter(ImageFilter.GaussianBlur(radius=1)) 
      image.save(self.out_dir + name.strip('g.jpg') + 'g.jpg')
    print('gaussian blur completed.')

  # rotation
  def rotation():
    print('performing rotation on the original images...')
    for name in img_names:
      image = Image.open(self.in_dir + name)
      for i in (-15, -10, 10, 15):
        image_r = image.rotate(i)
        image_r.save(self.out_dir + name[:len(name) - 4] + 'r' + str(i) + '.jpg')
    print('rotation completed.')

  #threshold
  def threshold():
    for name in img_names:
      data = np.array(Image.open(self.in_dir+name))
      index = data >=  128
      data[index] = 255
      index = data < 128
      data[index] = 0
      image = Image.fromarray(data)
      image.save(self.out_dir + name)
