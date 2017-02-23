import os
import sys
import re
import tables
import numpy as np

def main(argv):
    img_dir = "../data/chars_generated/"
    img_srcs = os.listdir(img_dir)
    is_jpg = re.compile(r'.+?\.jpg')
    print('generating class labels...') 
    if (len(img_srcs)>0):
        #hdf5_path = 'trainLabel.hdf5'
        #hdf5_file = tables.open_file(hdf5_path, mode='w')
        #filters = tables.Filters(complevel=5, complib='blosc')
        #trainLabel = hdf5_file.create_earray(hdf5_file.root, 'trainLabel',
        #                                   tables.Atom.from_dtype(np.dtype('Int32')), 
        #                                   shape=(0,1),
        #                                   filters=filters,
        #                                   expectedrows=919975)
        #
        counter = 0
        for img_src in img_srcs:
            if(is_jpg.match(img_src)):
               counter = counter + 1
               # if(counter%5000 == 0):
               #     print("processing the %dth picture..." %counter)
               if(counter > 20):
                   break
               print(img_src)
               # class_name = int(img_src.split('_')[0]) - 1
               # trainLabel.append(np.array(class_name).reshape((1,1)))
        # hdf5_file.close()

    print('class label generated.')
if __name__ == "__main__":
    main(sys.argv[1:])
