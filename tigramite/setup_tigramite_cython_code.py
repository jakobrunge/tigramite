from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
ext_modules = [
Extension("tigramite_cython_code", ["tigramite_cython_code.c"],
include_dirs=[numpy.get_include()]),
],
)