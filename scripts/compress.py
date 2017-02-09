import numpy as np
import tables
import sys

def main(argv):
    data_path = '../data/matrices.txt'
    features = open(data_path)
    totalData = features.readline().strip('\t').split('\t')
    totalData = np.asarray(totalData, dtype='float32').reshape((-1, 8*48*48))
    hdf5_path = 'testData.hdf5'
    hdf5_file = tables.open_file(hdf5_path, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc') 
    testData0 = hdf5_file.create_earray(hdf5_file.root, 'testData0', 
                                        tables.Atom.from_dtype(totalData.dtype), 
                                        shape=(0,),
                                        filters=filters,
                                        expectedrows=len(totalData))
    for n, row in enumerate(totalData):
        testData0.append(totalData[n])
    hdf5_file.close()


if __name__ == '__main__':
    main(sys.argv[1:])
