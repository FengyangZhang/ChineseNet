import numpy as np
import sys
import tables
from matplotlib import pyplot as plt
import img2txt
import Image
def main(argv):
    #hdf5_path = 'trainData_32.hdf5'
    #hdf5_file = tables.open_file(hdf5_path, mode='r')
    #hdf5_data = hdf5_file.root.trainData[919973]
    #hdf5_file.close()

    #hdf5_path = 'valData.hdf5'
    #hdf5_file = tables.open_file(hdf5_path, mode='w')
    #filters = tables.Filters(complevel=5, complib='blosc')
    #valData = hdf5_file.create_earray(hdf5_file.root, 'valData',
    #                                   tables.Atom.from_dtype(np.dtype('uint8')), 
    #                                   shape=(0, 18432),
    #                                   filters=filters)
    #valData.append(hdf5_data.reshape((1,18432)))
    img_src = '../data/chars_generated_32/999_31.jpg'
    img_mat = np.array(Image.open(img_src))    
    #print('the img size is %d' %img_mat.shape)
    directMap = img2txt.img2directMap(img_mat)
    print(directMap.shape)
    #print(hdf5_data)
    #print('going to plot the data for training...')
    #img_array = np.asarray(hdf5_data[48*48*3:48*48*4]).reshape((48,48))
    #plot_data(img_array)

    #hdf5_path = 'valData.hdf5'
    #hdf5_file = tables.open_file(hdf5_path, mode='r')
    #hdf5_data = hdf5_file.root.valData[0]
    #hdf5_file.close()
    #hdf5_data = hdf5_data.reshape((8,48,48)).transpose(1,2,0)
    #plot_data(hdf5_data[:,:,0])

def plot_data(data):
    plt.imshow(data, cmap='gray')
    plt.gca().axis('off')
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
