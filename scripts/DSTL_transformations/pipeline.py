#!/usr/bin/python
# coding: utf-8

from __future__ import with_statement

# This is intended to be run as a GIMP script
#from gimpfu import *

import sys, os, getopt
import numpy
import ift6266.data_generation.transformations.filetensor as ft
import random
import copy

# To debug locally, also call with -s 100 (to stop after ~100)
# (otherwise we allocate all needed memory, might be loonnng and/or crash
# if, lucky like me, you have an age-old laptop creaking from everywhere)
DEBUG = False
DEBUG_X = False
if DEBUG:
    DEBUG_X = False # Debug under X (pylab.show())

DEBUG_IMAGES_PATH = None
if DEBUG:
    # UNTESTED YET
    # To avoid loading NIST if you don't have it handy
    # (use with debug_images_iterator(), see main())
    # To use NIST, leave as = None
    DEBUG_IMAGES_PATH = None#'/home/francois/Desktop/debug_images'

# Directory where to dump images to visualize results
# (create it, otherwise it'll crash)
DEBUG_OUTPUT_DIR = 'debug_out'

DEFAULT_NIST_PATH = '/data/lisa/data/ift6266h10/train_data.ft'
DEFAULT_LABEL_PATH = '/data/lisa/data/ift6266h10/train_labels.ft'
DEFAULT_OCR_PATH = '/data/lisa/data/ocr_breuel/filetensor/unlv-corrected-2010-02-01-shuffled.ft'
DEFAULT_OCRLABEL_PATH = '/data/lisa/data/ocr_breuel/filetensor/unlv-corrected-2010-02-01-labels-shuffled.ft'
ARGS_FILE = os.environ['PIPELINE_ARGS_TMPFILE']

# PARSE COMMAND LINE ARGUMENTS
def get_argv():
    with open(ARGS_FILE) as f:
        args = [l.rstrip() for l in f.readlines()]
    return args

def usage():
    print '''
Usage: run_pipeline.sh [-m ...] [-z ...] [-o ...] [-p ...]
    -m, --max-complexity: max complexity to generate for an image
    -z, --probability-zero: probability of using complexity=0 for an image
    -o, --output-file: full path to file to use for output of images
    -p, --params-output-file: path to file to output params to
    -x, --labels-output-file: path to file to output labels to
    -f, --data-file: path to filetensor (.ft) data file (NIST)
    -l, --label-file: path to filetensor (.ft) labels file (NIST labels)
    -c, --ocr-file: path to filetensor (.ft) data file (OCR)
    -d, --ocrlabel-file: path to filetensor (.ft) labels file (OCR labels)
    -a, --prob-font: probability of using a raw font image
    -b, --prob-captcha: probability of using a captcha image
    -g, --prob-ocr: probability of using an ocr image
    -y, --seed: the job seed
    -t, --type: [default: 0:full transformations], 1:Nist-friendly transformations
    '''

try:
    opts, args = getopt.getopt(get_argv(), "r:m:z:o:p:x:s:f:l:c:d:a:b:g:y:t:", ["reload","max-complexity=", "probability-zero=", "output-file=", "params-output-file=", "labels-output-file=", 
"stop-after=", "data-file=", "label-file=", "ocr-file=", "ocrlabel-file=", "prob-font=", "prob-captcha=", "prob-ocr=", "seed=","type="])
except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        pdb.gimp_quit(0)
        sys.exit(2)

for o, a in opts:
    if o in ('-y','--seed'):
        random.seed(int(a))
        numpy.random.seed(int(a))

type_pipeline = 0
for o, a in opts:
    if o in ('-t','--type'):
        type_pipeline = int(a)

if DEBUG_X:
    import pylab
    pylab.ion()

from ift6266.data_generation.transformations.PoivreSel import PoivreSel
from ift6266.data_generation.transformations.thick import Thick
from ift6266.data_generation.transformations.BruitGauss import BruitGauss
from ift6266.data_generation.transformations.DistorsionGauss import DistorsionGauss
from ift6266.data_generation.transformations.PermutPixel import PermutPixel
from ift6266.data_generation.transformations.gimp_script import GIMP1
from ift6266.data_generation.transformations.Rature import Rature
from ift6266.data_generation.transformations.contrast import Contrast
from ift6266.data_generation.transformations.local_elastic_distortions import LocalElasticDistorter
from ift6266.data_generation.transformations.slant import Slant
from ift6266.data_generation.transformations.Occlusion import Occlusion
from ift6266.data_generation.transformations.add_background_image import AddBackground
from ift6266.data_generation.transformations.affine_transform import AffineTransformation
from ift6266.data_generation.transformations.ttf2jpg import ttf2jpg
from ift6266.data_generation.transformations.pycaptcha.Facade import generateCaptcha

if DEBUG:
    from visualizer import Visualizer
    # Either put the visualizer as in the MODULES_INSTANCES list
    # after each module you want to visualize, or in the
    # AFTER_EACH_MODULE_HOOK list (but not both, it's redundant)
    VISUALIZER = Visualizer(to_dir=DEBUG_OUTPUT_DIR,  on_screen=False)

###---------------------order of transformation module
if type_pipeline == 0:
    MODULE_INSTANCES = [Slant(),Thick(),AffineTransformation(),LocalElasticDistorter(),GIMP1(),Rature(),Occlusion(), PermutPixel(),DistorsionGauss(),AddBackground(), PoivreSel(), BruitGauss(), Contrast()]
    stop_idx = 0
if type_pipeline == 1:
    MODULE_INSTANCES = [Slant(),Thick(),AffineTransformation(),LocalElasticDistorter(),GIMP1(False),Rature(),Occlusion(), PermutPixel(),DistorsionGauss(),AddBackground(), PoivreSel(), BruitGauss(), Contrast()]
    stop_idx = 5
    #we disable transformation corresponding to MODULE_INSTANCES[stop_idx:] but we still need to apply them on dummy images
    #in order to be sure to have the same random generator state than with the default pipeline.
    #This is not optimal (we do more calculus than necessary) but it is a quick hack to produce similar results than previous generation



# These should have a "after_transform_callback(self, image)" method
# (called after each call to transform_image in a module)
AFTER_EACH_MODULE_HOOK = []
if DEBUG:
    AFTER_EACH_MODULE_HOOK = [VISUALIZER]

# These should have a "end_transform_callback(self, final_image" method
# (called after all modules have been called)
END_TRANSFORM_HOOK = []
if DEBUG:
    END_TRANSFORM_HOOK = [VISUALIZER]

class Pipeline():
    def __init__(self, modules, num_img, image_size=(32,32)):
        self.modules = modules
        self.num_img = num_img
        self.num_params_stored = 0
        self.image_size = image_size

        self.init_memory()

    def init_num_params_stored(self):
        # just a dummy call to regenerate_parameters() to get the
        # real number of params (only those which are stored)
        self.num_params_stored = 0
        for m in self.modules:
            self.num_params_stored += len(m.regenerate_parameters(0.0))

    def init_memory(self):
        self.init_num_params_stored()

        total = self.num_img
        num_px = self.image_size[0] * self.image_size[1]

        self.res_data = numpy.empty((total, num_px), dtype=numpy.uint8)
        # +1 to store complexity
        self.params = numpy.empty((total, self.num_params_stored+len(self.modules)))
        self.res_labels = numpy.empty(total, dtype=numpy.int32)

    def run(self, img_iterator, complexity_iterator):
        img_size = self.image_size

        should_hook_after_each = len(AFTER_EACH_MODULE_HOOK) != 0
        should_hook_at_the_end = len(END_TRANSFORM_HOOK) != 0

        for img_no, (img, label) in enumerate(img_iterator):
            sys.stdout.flush()
            
            global_idx = img_no
            
            img = img.reshape(img_size)

            param_idx = 0
            mod_idx = 0
            for mod in self.modules:
                # This used to be done _per batch_,
                # ie. out of the "for img" loop
                complexity = complexity_iterator.next()    
                #better to do a complexity sampling for each transformations in order to have more variability
                #otherwise a lot of images similar to the source are generated (i.e. when complexity is close to 0 (1/8 of the time))
                #we need to save the complexity of each transformations and the sum of these complexity is a good indicator of the overall
                #complexity
                self.params[global_idx, mod_idx] = complexity
                mod_idx += 1
                 
                p = mod.regenerate_parameters(complexity)
                self.params[global_idx, param_idx+len(self.modules):param_idx+len(p)+len(self.modules)] = p
                param_idx += len(p)
                
                if not(stop_idx) or stop_idx > mod_idx:  
                    img = mod.transform_image(img)
                else:
                    tmp = mod.transform_image(copy.copy(img)) 
                    #this is done to be sure to have the same global random generator state
                    #we don't apply the transformation on the original image but on a copy in case of in-place transformations

                if should_hook_after_each:
                    for hook in AFTER_EACH_MODULE_HOOK:
                        hook.after_transform_callback(img)

            self.res_data[global_idx] = \
                    img.reshape((img_size[0] * img_size[1],))*255
            self.res_labels[global_idx] = label

            if should_hook_at_the_end:
                for hook in END_TRANSFORM_HOOK:
                    hook.end_transform_callback(img)

    def write_output(self, output_file_path, params_output_file_path, labels_output_file_path):
        with open(output_file_path, 'wb') as f:
            ft.write(f, self.res_data)
        
        #if type_pipeline == 0: #only needed for type 0 pipeline
        numpy.save(params_output_file_path, self.params)
        
        with open(labels_output_file_path, 'wb') as f:
            ft.write(f, self.res_labels)
                

##############################################################################
# COMPLEXITY ITERATORS
# They're called once every img, to get the complexity to use for that img
# they must be infinite (should never throw StopIteration when calling next())

# probability of generating 0 complexity, otherwise
# uniform over 0.0-max_complexity
def range_complexity_iterator(probability_zero, max_complexity):
    assert max_complexity <= 1.0
    n = numpy.random.uniform(0.0, 1.0)
    n = 2.0 #hack to bug fix, having a min complexity is not necessary and we need the same seed...
    while True:
        if n < probability_zero:
            yield 0.0
        else:
            yield numpy.random.uniform(0.0, max_complexity)

##############################################################################
# DATA ITERATORS
# They can be used to interleave different data sources etc.

'''
# Following code (DebugImages and iterator) is untested

def load_image(filepath):
    _RGB_TO_GRAYSCALE = [0.3, 0.59, 0.11, 0.0]
    img = Image.open(filepath)
    img = numpy.asarray(img)
    if len(img.shape) > 2:
        img = (img * _RGB_TO_GRAYSCALE).sum(axis=2)
    return (img / 255.0).astype('float')

class DebugImages():
    def __init__(self, images_dir_path):
        import glob, os.path
        self.filelist = glob.glob(os.path.join(images_dir_path, "*.png"))

def debug_images_iterator(debug_images):
    for path in debug_images.filelist:
        yield load_image(path)
'''

class NistData():
    def __init__(self, nist_path, label_path, ocr_path, ocrlabel_path):
        self.train_data = open(nist_path, 'rb')
        self.train_labels = open(label_path, 'rb')
        self.dim = tuple(ft._read_header(self.train_data)[3])
        # in order to seek to the beginning of the file
        self.train_data.close()
        self.train_data = open(nist_path, 'rb')
        self.ocr_data = open(ocr_path, 'rb')
        self.ocr_labels = open(ocrlabel_path, 'rb')

# cet iterator load tout en ram
def nist_supp_iterator(nist, prob_font, prob_captcha, prob_ocr, num_img):
    img = ft.read(nist.train_data)
    labels = ft.read(nist.train_labels)
    if prob_ocr:
        ocr_img = ft.read(nist.ocr_data)
        ocr_labels = ft.read(nist.ocr_labels)
    ttf = ttf2jpg()
    L = [chr(ord('0')+x) for x in range(10)] + [chr(ord('A')+x) for x in range(26)] + [chr(ord('a')+x) for x in range(26)]

    for i in xrange(num_img):
        r = numpy.random.rand()
        if r <= prob_font:
            yield ttf.generate_image()
        elif r <=prob_font + prob_captcha:
            (arr, charac) = generateCaptcha(0,1)
            yield arr.astype(numpy.float32)/255, L.index(charac[0])
        elif r <= prob_font + prob_captcha + prob_ocr:
            j = numpy.random.randint(len(ocr_labels))
            yield ocr_img[j].astype(numpy.float32)/255, ocr_labels[j]
        else:
            j = numpy.random.randint(len(labels))
            yield img[j].astype(numpy.float32)/255, labels[j]


# Mostly for debugging, for the moment, just to see if we can
# reload the images and parameters.
def reload(output_file_path, params_output_file_path):
    images_ft = open(output_file_path, 'rb')
    images_ft_dim = tuple(ft._read_header(images_ft)[3])

    print "Images dimensions: ", images_ft_dim

    params = numpy.load(params_output_file_path)

    print "Params dimensions: ", params.shape
    print params
    

##############################################################################
# MAIN


# Might be called locally or through dbidispatch. In all cases it should be
# passed to the GIMP executable to be able to use GIMP filters.
# Ex: 
def _main():
    #global DEFAULT_NIST_PATH, DEFAULT_LABEL_PATH, DEFAULT_OCR_PATH, DEFAULT_OCRLABEL_PATH
    #global getopt, get_argv

    max_complexity = 0.5 # default
    probability_zero = 0.1 # default
    output_file_path = None
    params_output_file_path = None
    labels_output_file_path = None
    nist_path = DEFAULT_NIST_PATH
    label_path = DEFAULT_LABEL_PATH
    ocr_path = DEFAULT_OCR_PATH
    ocrlabel_path = DEFAULT_OCRLABEL_PATH
    prob_font = 0.0
    prob_captcha = 0.0
    prob_ocr = 0.0
    stop_after = None
    reload_mode = False

    for o, a in opts:
        if o in ('-m', '--max-complexity'):
            max_complexity = float(a)
            assert max_complexity >= 0.0 and max_complexity <= 1.0
        elif o in ('-r', '--reload'):
            reload_mode = True
        elif o in ("-z", "--probability-zero"):
            probability_zero = float(a)
            assert probability_zero >= 0.0 and probability_zero <= 1.0
        elif o in ("-o", "--output-file"):
            output_file_path = a
        elif o in ('-p', "--params-output-file"):
            params_output_file_path = a
        elif o in ('-x', "--labels-output-file"):
            labels_output_file_path = a
        elif o in ('-s', "--stop-after"):
            stop_after = int(a)
        elif o in ('-f', "--data-file"):
            nist_path = a
        elif o in ('-l', "--label-file"):
            label_path = a
        elif o in ('-c', "--ocr-file"):
            ocr_path = a
        elif o in ('-d', "--ocrlabel-file"):
            ocrlabel_path = a
        elif o in ('-a', "--prob-font"):
            prob_font = float(a)
        elif o in ('-b', "--prob-captcha"):
            prob_captcha = float(a)
        elif o in ('-g', "--prob-ocr"):
            prob_ocr = float(a)
        elif o in ('-y', "--seed"):
            pass
        elif o in ('-t', "--type"):
            pass            
        else:
            assert False, "unhandled option"

    if output_file_path == None or params_output_file_path == None or labels_output_file_path == None:
        print "Must specify the three output files."
        usage()
        pdb.gimp_quit(0)
        sys.exit(2)

    if reload_mode:
        reload(output_file_path, params_output_file_path)
    else:
        if DEBUG_IMAGES_PATH:
            '''
            # This code is yet untested
            debug_images = DebugImages(DEBUG_IMAGES_PATH)
            num_img = len(debug_images.filelist)
            pl = Pipeline(modules=MODULE_INSTANCES, num_img=num_img, image_size=(32,32))
            img_it = debug_images_iterator(debug_images)
            '''
        else:
            nist = NistData(nist_path, label_path, ocr_path, ocrlabel_path)
            num_img = 819200 # 800 Mb file
            if stop_after:
                num_img = stop_after
            pl = Pipeline(modules=MODULE_INSTANCES, num_img=num_img, image_size=(32,32))
            img_it = nist_supp_iterator(nist, prob_font, prob_captcha, prob_ocr, num_img)

        cpx_it = range_complexity_iterator(probability_zero, max_complexity)
        pl.run(img_it, cpx_it)
        pl.write_output(output_file_path, params_output_file_path, labels_output_file_path)

try:
    _main()
except:
    print "Unexpected error"

if DEBUG_X:
    pylab.ioff()
    pylab.show()

pdb.gimp_quit(0)

