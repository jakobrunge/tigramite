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
CMDCLASS = {'build_ext': UseNumpyHeadersBuildExt}

# Define the minimal classes needed to install and run tigramite
INSTALL_REQUIRES = ["numpy", "scipy", "sklearn"]

# Define the extras needed
EXTRAS_REQUIRE = {
    'Gaussian Process (GP) Regression':  ['sklearn>=0.18'],
    'R-based ACE, also requires acepack installed in R':  ['rpy2'],
    'pure-python ACE':  ['ace>=0.3'],
    'plotting': ['matplotlib>=1.5', 'networkx>=1.10'],
    'p-value corrections': ['statsmodels']
}

# Define the packages needed for testing
TESTS_REQUIRE = ['nose']

# Define the external modules to build
EXT_MODULES = [Extension("tigramite.tigramite_cython_code",
                         ["tigramite/tigramite_cython_code.c"])]

# Run the setup
setup(
    name='tigramite',
    version='3.0.0-beta',
    packages=['tigramite'],
    license='GNU General Public License v3.0',
    description='Tigramite causal discovery for time series',
    author='Jakob Runge',
    author_email='jakobrunge@posteo.de',
    url='https://github.com/jakobrunge/tigramite/',
    long_description=io.open('README.md', 'r', encoding='utf-8').read(),
    keywords='causality time-series',
    cmdclass=CMDCLASS,
    ext_modules=EXT_MODULES,
    install_requires=INSTALL_REQUIRES,
    test_suite='nose.collector',
    tests_require=TESTS_REQUIRE,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License '\
            ':: OSI Approved '\
            ':: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 2.7',
    ],
)
