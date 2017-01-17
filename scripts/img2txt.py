import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import re

def main(argv):
    img_dir = "madeups_gen/"
    img_srcs = os.listdir(img_dir)
    file=open('matrices.txt','w')
    # leave files that are not jpg files
    is_jpg = re.compile(r'.+?\.jpg')

    print('generating matrices...')
    if (len(img_srcs)>0):
        for img_src in img_srcs:
            if(is_jpg.match(img_src)):
                img_src = img_dir + img_src
                img_mat = np.array(Image.open(img_src))
                # if(img_mat.shape != (28, 18)):
                #     print(img_src)
                #     return
                img_row = '\t'.join('\t'.join('%d' %x for x in y) for y in img_mat)
                file.write(img_row)
                file.write('\t')
    file = open('matrices.txt')
    img_row = file.readline().strip('\t').split('\t')
    img_mat = np.asarray(img_row, dtype='int32')
    print(img_mat.shape)
    img_mat = np.reshape(img_mat, (-1, 28, 18))
    print('matrices generated.')
    plt.imshow(img_mat[0])
    plt.show()
if __name__ == '__main__':
    main(sys.argv[1:])