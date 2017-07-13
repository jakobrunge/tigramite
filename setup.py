# from distutils.core import setup, Extension
from setuptools import setup, Extension

import numpy

import io

# If cython is available, the included cython *.pyx file
# is compiled, otherwise the *.c file is used
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True


install_requires = [
    'numpy',
    'scipy'
]

extras_require = {
    'Gaussian Process (GP) Regression':  ['sklearn>=0.18'],
    'R-based ACE, also requires acepack installed in R':  ['rpy2'],
    'pure-python ACE':  ['ace>=0.3'],
    'plotting': ['matplotlib>=1.5', 'networkx>=1.10'],
    'p-value corrections': ['statsmodels']
}

tests_require = ['nose']


cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += [
        Extension("tigramite.tigramite_cython_code", 
                  [ "tigramite/tigramite_cython_code.pyx" ],
                  include_dirs=[numpy.get_include()]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("tigramite.tigramite_cython_code", 
                  ["tigramite/tigramite_cython_code.c"],
                  include_dirs=[numpy.get_include()]),
    ]

setup(
    name='tigramite',
    version='3.0b0',
    packages=['tigramite',],
    license='GNU General Public License v3.0',
    description='Tigramite causal discovery for time series',
    author='Jakob Runge',
    author_email='jakobrunge@posteo.de',
    url='https://github.com/jakobrunge/tigramite/',
    long_description=io.open('README.md', 'r', encoding='utf-8').read(),
    keywords = 'causality time-series',
    cmdclass = cmdclass,
    ext_modules=ext_modules,
    install_requires=install_requires,
    test_suite = 'nose.collector',
    tests_require = tests_require,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 2.7',
    ],
)