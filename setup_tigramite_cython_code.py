from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


# setup(
#     ext_modules = cythonize("tigramite/tigramite_cython_code.pyx"), 
#     include_dirs=[numpy.get_include()]
# )


setup(
ext_modules = [
Extension("tigramite/tigramite_cython_code", ["tigramite/tigramite_cython_code.c"],
include_dirs=[numpy.get_include()]),
],
)