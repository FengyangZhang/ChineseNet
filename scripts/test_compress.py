import numpy as np
import sys
import tables
from matplotlib import pyplot as plt
def main(argv):
    hdf5_path = 'testData.hdf5'
    hdf5_file = tables.open_file(hdf5_path, mode='r')
    hdf5_data = hdf5_file.root.testData0[:]
    print('going to plot the data for testing...')
    img_array = np.asarray(hdf5_data[48*48*3:48*48*4]).reshape((48,48))
    plot_data(img_array)

def plot_data(data):
    plt.imshow(data, cmap='gray')
    plt.gca().axis('off')
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
