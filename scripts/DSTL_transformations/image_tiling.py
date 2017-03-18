"""
Illustrate filters (or data) in a grid of small image-shaped tiles.

Note: taken from the pylearn codebase on Feb 4, 2010 (fsavard)
"""

import numpy
from PIL import Image

def scale_to_unit_interval(ndar,eps=1e-8):
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max()+eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0,0),
        scale_rows_to_unit_interval=True, 
        output_pixel_vals=True
        ):
    """
    Transform an array with one flattened image per row, into an array in which images are
    reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images, and also columns of
    matrices for transforming those rows (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can be 2-D ndarrays or None
    :param X: a 2-D array in which every row is a flattened image.
    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image
    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :returns: array suitable for viewing as an image.  (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp 
        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0,0,0,255]
        else:
            channel_defaults = [0.,0.,0.,1.]

        for i in xrange(4):
            if X[i] is None:
                out_array[:,:,i] = numpy.zeros(out_shape,
                        dtype='uint8' if output_pixel_vals else out_array.dtype
                        )+channel_defaults[i]
            else:
                out_array[:,:,i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        H, W = img_shape
        Hs, Ws = tile_spacing

        out_array = numpy.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)
        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    out_array[
                        tile_row * (H+Hs):tile_row*(H+Hs)+H,
                        tile_col * (W+Ws):tile_col*(W+Ws)+W
                        ] \
                        = this_img * (255 if output_pixel_vals else 1)
        return out_array


