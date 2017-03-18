#!/usr/bin/python
# coding: utf-8

'''
Implementation of elastic distortions as described in
Simard, Steinkraus, Platt, "Best Practices for Convolutional
    Neural Networks Applied to Visual Document Analysis", 2003

Author: François Savard
Date: Fall 2009, revised Winter 2010

Usage: create the Distorter with proper alpha, sigma etc.
    Then each time you want to change the distortion field applied,
    call regenerate_field(). 

    (The point behind this is that regeneration takes some time,
    so we better reuse the fields a few times)
'''

import sys
import math
import numpy
import numpy.random
import scipy.signal # convolve2d

_TEST_DIR = "/u/savardf/ift6266/debug_images/"

def _raw_zeros(size):
    return [[0 for i in range(size[1])] for j in range(size[0])]

class ElasticDistortionParams():
    def __init__(self, image_size=(32,32), alpha=0.0, sigma=0.0):
        self.image_size = image_size
        self.alpha = alpha
        self.sigma = sigma

        h,w = self.image_size

        self.matrix_tl_corners_rows = _raw_zeros((h,w))
        self.matrix_tl_corners_cols = _raw_zeros((h,w))

        self.matrix_tr_corners_rows = _raw_zeros((h,w))
        self.matrix_tr_corners_cols = _raw_zeros((h,w))

        self.matrix_bl_corners_rows = _raw_zeros((h,w))
        self.matrix_bl_corners_cols = _raw_zeros((h,w))

        self.matrix_br_corners_rows = _raw_zeros((h,w))
        self.matrix_br_corners_cols = _raw_zeros((h,w))

        # those will hold the precomputed ratios for
        # bilinear interpolation
        self.matrix_tl_multiply = numpy.zeros((h,w))
        self.matrix_tr_multiply = numpy.zeros((h,w))
        self.matrix_bl_multiply = numpy.zeros((h,w))
        self.matrix_br_multiply = numpy.zeros((h,w))

    def alpha_sigma(self):
        return [self.alpha, self.sigma]

class LocalElasticDistorter():
    def __init__(self, image_size=(32,32)):
        self.image_size = image_size

        self.current_complexity_10 = 0
        self.current_complexity = 0

        # number of precomputed fields
        # (principle: as complexity doesn't change often, we can
        # precompute a certain number of fields for a given complexity,
        # each with its own parameters. That way, we have good
        # randomization, but we're much faster).
        self.to_precompute_per_complexity = 50

        # Both use ElasticDistortionParams
        self.current_params = None
        self.precomputed_params = [[] for i in range(10)]

        # 
        self.kernel_size = None
        self.kernel = None

        # set some defaults
        self.regenerate_parameters(0.0)

    def get_settings_names(self):
        return []

    def _floor_complexity(self, complexity):
        return self._to_complexity_10(complexity) / 10.0

    def _to_complexity_10(self, complexity):
        return min(9, max(0, int(complexity * 10)))

    def regenerate_parameters(self, complexity):
        complexity_10 = self._to_complexity_10(complexity)

        if complexity_10 != self.current_complexity_10:
            self.current_complexity_10 = complexity_10
            self.current_complexity = self._floor_complexity(complexity)

        if len(self.precomputed_params[complexity_10]) <= self.to_precompute_per_complexity:
            # not yet enough params generated, produce one more
            # and append to list
            new_params = self._initialize_new_params()
            new_params = self._generate_fields(new_params)
            self.current_params = new_params
            self.precomputed_params[complexity_10].append(new_params)
        else:
            # if we have enough precomputed fields, just select one
            # at random and set parameters to match what they were
            # when the field was generated
            idx = numpy.random.randint(0, len(self.precomputed_params[complexity_10]))
            self.current_params = self.precomputed_params[complexity_10][idx]

        # don't return anything, to avoid storing deterministic parameters
        return [] # self.current_params.alpha_sigma()

    def get_parameters_determined_by_complexity(self, complexity):
        tmp_params = self._initialize_new_params(_floor_complexity(complexity))
        return tmp_params.alpha_sigma()

    def get_settings_names_determined_by_complexity(self, complexity):
        return ['alpha', 'sigma']

    # adapted from http://blenderartists.org/forum/showthread.php?t=163361
    def _gen_gaussian_kernel(self, sigma):
        # the kernel size can change DRAMATICALLY the time 
        # for the blur operation... so even though results are better
        # with a bigger kernel, we need to compromise here
        # 1*s is very different from 2*s, but there's not much difference
        # between 2*s and 4*s
        ks = self.kernel_size
        s = sigma
        target_ks = (1.5*s, 1.5*s)
        if not ks is None and ks[0] == target_ks[0] and ks[1] == target_ks[1]:
            # kernel size is good, ok, no need to regenerate
            return
        self.kernel_size = target_ks
        h,w = self.kernel_size
        a,b = h/2.0, w/2.0
        y,x = numpy.ogrid[0:w, 0:h]
        gauss = numpy.exp(-numpy.square((x-a)/s))*numpy.exp(-numpy.square((y-b)/s))
        # Normalize so we don't reduce image intensity
        self.kernel = gauss/gauss.sum()

    def _gen_distortion_field(self, params):
        self._gen_gaussian_kernel(params.sigma)

        # we add kernel_size on all four sides so blurring
        # with the kernel produces a smoother result on borders
        ks0 = self.kernel_size[0]
        ks1 = self.kernel_size[1]
        sz0 = self.image_size[1] + ks0
        sz1 = self.image_size[0] + ks1
        field = numpy.random.uniform(-1.0, 1.0, (sz0, sz1))
        field = scipy.signal.convolve2d(field, self.kernel, mode='same')

        # crop only image_size in the middle
        field = field[ks0:ks0+self.image_size[0], ks1:ks1+self.image_size[1]]

        return params.alpha * field
        

    def _initialize_new_params(self, complexity=None):
        if not complexity:
            complexity = self.current_complexity

        params = ElasticDistortionParams(self.image_size)

        # pour faire progresser la complexité un peu plus vite
        # tout en gardant les extrêmes de 0.0 et 1.0
        complexity = complexity ** (1./3.)

        # the smaller the alpha, the closest the pixels are fetched
        # a max of 10 is reasonable
        params.alpha = complexity * 10.0

        # the bigger the sigma, the smoother is the distortion
        # max of 1 is "reasonable", but produces VERY noisy results
        # And the bigger the sigma, the bigger the blur kernel, and the
        # slower the field generation, btw.
        params.sigma = 10.0 - (7.0 * complexity)

        return params

    def _generate_fields(self, params):
        '''
        Here's how the code works:
        - We first generate "distortion fields" for x and y with these steps:
            - Uniform noise over [-1, 1] in a matrix of size (h,w)
            - Blur with a Gaussian kernel of spread sigma
            - Multiply by alpha
        - Then (conceptually) to compose the distorted image, we loop over each pixel
            of the new image and use the corresponding x and y distortions
            (from the matrices generated above) to identify pixels
            of the old image from which we fetch color data. As the
            coordinates are not integer, we interpolate between the
            4 nearby pixels (top left, top right etc.).
        - That's just conceptually. Here I'm using matrix operations
            to speed up the computation. I first identify the 4 nearby
            pixels in the old image for each pixel in the distorted image.
            I can then use them as "fancy indices" to extract the proper
            pixels for each new pixel.
        - Then I multiply those extracted nearby points by precomputed
            ratios for the bilinear interpolation.
        '''

        p = params

        dist_fields = [None, None]
        dist_fields[0] = self._gen_distortion_field(params)
        dist_fields[1] = self._gen_distortion_field(params)

        #pylab.imshow(dist_fields[0])
        #pylab.show()

        # regenerate distortion index matrices
        # "_rows" are row indices
        # "_cols" are column indices
        # (separated due to the way fancy indexing works in numpy)
        h,w = p.image_size

        for y in range(h):
            for x in range(w): 
                distort_x = dist_fields[0][y,x]
                distort_y = dist_fields[1][y,x]

                # the "target" is the coordinate we fetch color data from
                # (in the original image)
                # target_left and _top are the rounded coordinate on the
                # left/top of this target (float) coordinate
                target_pixel = (y+distort_y, x+distort_x)

                target_left = int(math.floor(x + distort_x))
                target_top = int(math.floor(y + distort_y))

                index_tl = [target_top, target_left]
                index_tr = [target_top, target_left+1]
                index_bl = [target_top+1, target_left]
                index_br = [target_top+1, target_left+1]

                # x_ratio is the ratio of importance of left pixels
                # y_ratio is the """" of top pixels
                # (in bilinear combination)
                y_ratio = 1.0 - (target_pixel[0] - target_top)
                x_ratio = 1.0 - (target_pixel[1] - target_left)

                # We use a default background color of 0 for displacements
                # outside of boundaries of the image.

                # if top left outside bounds
                if index_tl[0] < 0 or index_tl[0] >= h or index_tl[1] < 0 or index_tl[1] >= w: 
                    p.matrix_tl_corners_rows[y][x] = 0
                    p.matrix_tl_corners_cols[y][x] = 0
                    p.matrix_tl_multiply[y,x] = 0
                else:
                    p.matrix_tl_corners_rows[y][x] = index_tl[0]
                    p.matrix_tl_corners_cols[y][x] = index_tl[1]
                    p.matrix_tl_multiply[y,x] = x_ratio*y_ratio

                # if top right outside bounds
                if index_tr[0] < 0 or index_tr[0] >= h or index_tr[1] < 0 or index_tr[1] >= w:
                    p.matrix_tr_corners_rows[y][x] = 0
                    p.matrix_tr_corners_cols[y][x] = 0
                    p.matrix_tr_multiply[y,x] = 0
                else:
                    p.matrix_tr_corners_rows[y][x] = index_tr[0]
                    p.matrix_tr_corners_cols[y][x] = index_tr[1]
                    p.matrix_tr_multiply[y,x] = (1.0-x_ratio)*y_ratio

                # if bottom left outside bounds
                if index_bl[0] < 0 or index_bl[0] >= h or index_bl[1] < 0 or index_bl[1] >= w:
                    p.matrix_bl_corners_rows[y][x] = 0
                    p.matrix_bl_corners_cols[y][x] = 0
                    p.matrix_bl_multiply[y,x] = 0
                else:
                    p.matrix_bl_corners_rows[y][x] = index_bl[0]
                    p.matrix_bl_corners_cols[y][x] = index_bl[1]
                    p.matrix_bl_multiply[y,x] = x_ratio*(1.0-y_ratio)

                # if bottom right outside bounds
                if index_br[0] < 0 or index_br[0] >= h or index_br[1] < 0 or index_br[1] >= w:
                    p.matrix_br_corners_rows[y][x] = 0
                    p.matrix_br_corners_cols[y][x] = 0
                    p.matrix_br_multiply[y,x] = 0
                else:
                    p.matrix_br_corners_rows[y][x] = index_br[0]
                    p.matrix_br_corners_cols[y][x] = index_br[1]
                    p.matrix_br_multiply[y,x] = (1.0-x_ratio)*(1.0-y_ratio)

        # not really necessary, but anyway
        return p

    def transform_image(self, image):
        p = self.current_params

        # index pixels to get the 4 corners for bilinear combination
        tl_pixels = image[p.matrix_tl_corners_rows, p.matrix_tl_corners_cols]
        tr_pixels = image[p.matrix_tr_corners_rows, p.matrix_tr_corners_cols]
        bl_pixels = image[p.matrix_bl_corners_rows, p.matrix_bl_corners_cols]
        br_pixels = image[p.matrix_br_corners_rows, p.matrix_br_corners_cols]

        # bilinear ratios, elemwise multiply
        tl_pixels = numpy.multiply(tl_pixels, p.matrix_tl_multiply)
        tr_pixels = numpy.multiply(tr_pixels, p.matrix_tr_multiply)
        bl_pixels = numpy.multiply(bl_pixels, p.matrix_bl_multiply)
        br_pixels = numpy.multiply(br_pixels, p.matrix_br_multiply)

        # sum to finish bilinear combination
        return numpy.sum([tl_pixels,tr_pixels,bl_pixels,br_pixels], axis=0).astype(numpy.float32)

# TESTS ----------------------------------------------------------------------

def _load_image(filepath):
    _RGB_TO_GRAYSCALE = [0.3, 0.59, 0.11, 0.0]
    img = Image.open(filepath)
    img = numpy.asarray(img)
    if len(img.shape) > 2:
        img = (img * _RGB_TO_GRAYSCALE).sum(axis=2)
    return (img / 255.0).astype('float')

def _specific_test():
    imgpath = os.path.join(_TEST_DIR, "d.png")
    img = _load_image(imgpath)
    dist = LocalElasticDistorter((32,32))
    print dist.regenerate_parameters(0.5)
    img = dist.transform_image(img)
    print dist.get_parameters_determined_by_complexity(0.4)
    pylab.imshow(img)
    pylab.show()

def _complexity_tests():
    imgpath = os.path.join(_TEST_DIR, "d.png")
    dist = LocalElasticDistorter((32,32))
    orig_img = _load_image(imgpath)
    html_content = '''<html><body>Original:<br/><img src='d.png'>'''
    for complexity in numpy.arange(0.0, 1.1, 0.1):
        html_content += '<br/>Complexity: ' + str(complexity) + '<br/>'
        for i in range(10):
            t1 = time.time()
            dist.regenerate_parameters(complexity)
            t2 = time.time()
            print "diff", t2-t1
            img = dist.transform_image(orig_img)
            filename = "complexity_" + str(complexity) + "_" + str(i) + ".png"
            new_path = os.path.join(_TEST_DIR, filename)
            _save_image(img, new_path)
            html_content += '<img src="' + filename + '">'
    html_content += "</body></html>"
    html_file = open(os.path.join(_TEST_DIR, "complexity.html"), "w")
    html_file.write(html_content)
    html_file.close()
    
def _complexity_benchmark():
    imgpath = os.path.join(_TEST_DIR, "d.png")
    dist = LocalElasticDistorter((32,32))
    orig_img = _load_image(imgpath)

    for cpx in (0.21, 0.35):
        # time the first 10
        t1 = time.time()
        for i in range(10):
            dist.regenerate_parameters(cpx)
            img = dist.transform_image(orig_img)
        t2 = time.time()

        print "first 10, total = ", t2-t1, ", avg=", (t2-t1)/10

        # time the next 40
        t1 = time.time()
        for i in range(40):
            dist.regenerate_parameters(cpx)
            img = dist.transform_image(orig_img)
        t2 = time.time()
       
        print "next 40, total = ", t2-t1, ", avg=", (t2-t1)/40

        # time the next 50
        t1 = time.time()
        for i in range(50):
            dist.regenerate_parameters(cpx)
            img = dist.transform_image(orig_img)
        t2 = time.time()
       
        print "next 50, total = ", t2-t1, ", avg=", (t2-t1)/50

        # time the next 1000 
        t1 = time.time()
        for i in range(1000):
            dist.regenerate_parameters(cpx)
            img = dist.transform_image(orig_img)
        t2 = time.time()
       
        print "next 1000, total = ", t2-t1, ", avg=", (t2-t1)/1000

    # time the next 1000 with old complexity
    t1 = time.time()
    for i in range(1000):
        dist.regenerate_parameters(0.21)
        img = dist.transform_image(orig_img)
    t2 = time.time()
   
    print "next 1000, total = ", t2-t1, ", avg=", (t2-t1)/1000




def _save_image(img, path):
    img2 = Image.fromarray((img * 255).astype('uint8'), "L")
    img2.save(path)

# TODO: reformat to follow new class... it function of complexity now
'''
def _distorter_tests():
    #import pylab
    #pylab.imshow(img)
    #pylab.show()

    for letter in ("d", "a", "n", "o"):
        img = _load_image("tests/" + letter + ".png")
        for alpha in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0):
            for sigma in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0):
                id = LocalElasticDistorter((32,32))
                img2 = id.distort_image(img)
                img2 = Image.fromarray((img2 * 255).astype('uint8'), "L")
                img2.save("tests/"+letter+"_alpha"+str(alpha)+"_sigma"+str(sigma)+".png")
'''

def _benchmark():
    img = _load_image("tests/d.png")
    dist = LocalElasticDistorter((32,32))
    dist.regenerate_parameters(0.0)
    import time
    t1 = time.time()
    for i in range(10000):
        if i % 1000 == 0:
            print "-"
        dist.distort_image(img)
    t2 = time.time()
    print "t2-t1", t2-t1
    print "avg", 10000/(t2-t1)

if __name__ == '__main__':
    import time
    import pylab
    import Image
    import os.path
    #_distorter_tests()
    #_benchmark()
    #_specific_test()
    #_complexity_tests()
    _complexity_benchmark()
    


