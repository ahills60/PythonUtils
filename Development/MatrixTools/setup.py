#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension

#Cython can be disabled using an environment variable
import os, os.path
try:
    enable_cython = not (os.environ['K_ENABLE_CYTHON'] in ['0', 'False'])
except KeyError:
    enable_cython = True

# Make this usable by people who don't have cython installed
if not(enable_cython):
    has_cython = False
else:
    try:
        from Cython.Distutils import build_ext
        has_cython = True
    except ImportError:
        has_cython = False

import numpy

# Define a cython-based extension module, using the generated sources if
# cython is not available.
pyxfile = 'matrixtools.pyx'
if has_cython and os.path.exists(pyxfile):
    pyx_sources = [pyxfile, 'c_numpy.pxd']
    cmdclass    = {'build_ext': build_ext}
else:
    pyx_sources = ['matrixtools.c']
    cmdclass    = {}


pyx_ext = Extension('matrixtools',
                 pyx_sources,
                 include_dirs = [numpy.get_include()])

# Call the routine which does the real work
setup(name        = 'matrixtools',
      version     = '0.1',
      author      = 'Andrew Hills',
      author_email= 'a.hills@sheffield.ac.uk',
      description = 'Matrix Tools using Cython',
      ext_modules = [pyx_ext],
      cmdclass    = cmdclass,
      )
