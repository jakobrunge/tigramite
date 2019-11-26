"""
Install tigramite
"""
from __future__ import print_function
import io
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Handle building against numpy headers before installing numpy
class UseNumpyHeadersBuildExt(build_ext):
    """
    Subclassed build_ext command.
    Allows for numpy to be imported after it is automatically installed.
    This lets us use numpy.get_include() while listing numpy as a needed
    dependency.
    """
    def run(self):
        # Import numpy here, only when headers are needed
        import numpy
        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())
        # Call original build_ext command
        build_ext.run(self)

# Handle cythonizing code only in development mode
def define_extension(extension_name, source_files=None):
    """
    Will define an extension from the *.c files unless in "setup.py develop"
    is called.  If this is in develop mode, then it tries to import cython
    and regenerate the *.c files from the *.pyx files
    :return: single-element list of needed extension
    """
    # Default source file
    if source_files is None:
        source_files = [extension_name.replace(".", "/") + ".c"]
    # If we are, try to import and use cythonize
    try:
        from Cython.Build import cythonize
        # Replace any extension in the source file list with .pyx
        source_files = [".".join(f.split(".")[:-1] + ["pyx"]) \
                        for f in source_files]
        # Return the cythonized extension
        return cythonize(Extension(extension_name, source_files))
    except ImportError:
        print("Cython cannot be found.  Skipping generation of C code from"+\
              " cython and using pre-compiled C code instead")
        return [Extension(extension_name, source_files)]

# Define the minimal classes needed to install and run tigramite
INSTALL_REQUIRES = ["numpy", "scipy", "six"]
# Define the all the possible extras needed
EXTRAS_REQUIRE = {
    'all' : ['scikit-learn>=0.18',#Gaussian Process (GP) Regression
             'matplotlib>=1.5',   #plotting
             'networkx>=1.10',    #plotting
             'rpy2'],             #R-based RCOT
    'R'   : ['rpy2']              #R-based RCOT
    }
# Define the packages needed for testing
TESTS_REQUIRE = ['nose',
                 'pytest',
                 'scikit-learn>=0.18',
                 'rpy2']
EXTRAS_REQUIRE['test'] = TESTS_REQUIRE
# Define the extras needed for development
EXTRAS_REQUIRE['dev'] = EXTRAS_REQUIRE['all'] + TESTS_REQUIRE + ['cython']

# Use a custom build to handle numpy.include_dirs() when building
CMDCLASS = {'build_ext': UseNumpyHeadersBuildExt}
# Define the external modules to build
EXT_MODULES = []
EXT_MODULES += define_extension("tigramite.tigramite_cython_code")

# Run the setup
setup(
    name='tigramite',
    version='4.1.0',
    packages=['tigramite'],
    license='GNU General Public License v3.0',
    description='Tigramite causal discovery for time series',
    author='Jakob Runge',
    author_email='jakob@jakob-runge.com',
    url='https://github.com/jakobrunge/tigramite/',
    long_description=io.open('README.md', 'r', encoding='utf-8').read(),
    keywords='causality, time-series',
    cmdclass=CMDCLASS,
    ext_modules=EXT_MODULES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    test_suite='tests',
    tests_require=TESTS_REQUIRE,
    classifiers=[
        'Development Status :: 4',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License '\
            ':: OSI Approved '\
            ':: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python',
    ]
)
