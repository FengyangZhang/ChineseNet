# This script is to test your modules to see if they conform to the module API
# defined on the wiki.
import random, numpy, gc, time, math, sys

# this is an example module that does stupid image value shifting

class DummyModule(object):
    def get_settings_names(self):
        return ['value']
    
    def regenerate_parameters(self, complexity):
        self._value = random.gauss(0, 0.5*complexity)
        return [self._value]

    def transform_image(self, image):
        return numpy.clip(image+self._value, 0, 1)
    
#import <your module>

# instanciate your class here (rather than DummyModule)
mod = DummyModule()

def error(msg):
    print "ERROR:", msg
    sys.exit(1)

def warn(msg):
    print "WARNING:", msg

def timeit(f, lbl):

    gc.disable()
    t = time.time()
    f()
    est = time.time() - t
    gc.enable()

    loops = max(1, int(10**math.floor(math.log(10/est, 10))))

    gc.disable()
    t = time.time()
    for _ in xrange(loops):
        f()

    print lbl, "(", loops, "loops ):", (time.time() - t)/loops, "s"
    gc.enable()

########################
# get_settings_names() #
########################

print "Testing get_settings_names()"

names = mod.get_settings_names()

if type(names) is not list:
    error("Must return a list")

if not all(type(e) is str for e in names):
    warn("The elements of the list should be strings")

###########################
# regenerate_parameters() #
###########################

print "Testing regenerate_parameters()"

params = mod.regenerate_parameters(0.2)

if type(params) is not list:
    error("Must return a list")

if len(params) != len(names):
    error("the returned parameter list must have the same length as the number of parameters")

params2 = mod.regenerate_parameters(0.2)
if len(names) != 0 and params == params2:
    error("the complexity parameter determines the distribution of the parameters, not their value")

mod.regenerate_parameters(0.0)
mod.regenerate_parameters(1.0)
    
mod.regenerate_parameters(0.5)

#####################
# transform_image() #
#####################

print "Testing transform_image()"

imgr = numpy.random.random_sample((32, 32)).astype(numpy.float32)
img1 = numpy.ones((32, 32), dtype=numpy.float32)
img0 = numpy.zeros((32, 32), dtype=numpy.float32)

resr = mod.transform_image(imgr)

if type(resr) is not numpy.ndarray:
    error("Must return an ndarray")

if resr.shape != (32, 32):
    error("Must return 32x32 array")

if resr.dtype != numpy.float32:
    error("Must return float32 array")

res1 = mod.transform_image(img1)
res0 = mod.transform_image(img0)

if res1.max() > 1.0 or res0.max() > 1.0:
    error("Must keep array values between 0 and 1")

if res1.min() < 0.0 or res0.min() < 0.0:
    error("Must keep array values between 0 and 1")

mod.regenerate_parameters(0.0)
mod.transform_image(imgr)
mod.regenerate_parameters(1.0)
mod.transform_image(imgr)

print "Bonus Stage: timings"

timeit(lambda: None, "empty")
timeit(lambda: mod.regenerate_parameters(0.5), "regenerate_parameters()")
timeit(lambda: mod.transform_image(imgr), "tranform_image()")

def f():
    mod.regenerate_parameters(0.2)
    mod.transform_image(imgr)

timeit(f, "regen and transform")
