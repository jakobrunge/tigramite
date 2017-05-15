from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("tigramite_cython_code.pyx"),
    include_dirs=[numpy.get_include()]
)