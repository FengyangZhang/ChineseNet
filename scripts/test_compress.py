import numpy as np
import sys
import tables
from matplotlib import pyplot as plt
def main(argv):
    hdf5_path = 'trainData.hdf5'
    hdf5_file = tables.open_file(hdf5_path, mode='r')
    hdf5_data = hdf5_file.root.trainData[0]
    hdf5_file.close()

    hdf5_path = 'valData.hdf5'
    hdf5_file = tables.open_file(hdf5_path, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    valData = hdf5_file.create_earray(hdf5_file.root, 'valData',
                                       tables.Atom.from_dtype(np.dtype('Int8')), 
                                       shape=(0, 18432),
                                       filters=filters)
    valData.append(hdf5_data.reshape((1,18432)))
    #print('the data size is %d' %hdf5_data.shape)
    #print(hdf5_data)
    #print('going to plot the data for training...')
    #img_array = np.asarray(hdf5_data[48*48*3:48*48*4]).reshape((48,48))
    #plot_data(img_array)

def plot_data(data):
    plt.imshow(data, cmap='gray')
    plt.gca().axis('off')
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])