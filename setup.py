"""
Install tigramite
"""
from __future__ import print_function
import pathlib
import os
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
        self.distribution.fetch_build_eggs(["numpy"])
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
        source_files = [str((pathlib.Path(__file__).parent / extension_name.replace(".", "/")).with_suffix(".c"))]
    # If we are, try to import and use cythonize
    try:
        from Cython.Build import cythonize
        # Return the cythonized extension
        pyx_path = str((pathlib.Path(__file__).parent / extension_name.replace(".", "/")).with_suffix(".pyx"))
        return cythonize([pyx_path], language_level = "3")
    except ImportError:
        print(
            "Cython cannot be found. Skipping generation of C code from"
            + " cython and using pre-compiled C code instead"
        )
        return [Extension(extension_name, source_files, 
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'],)]



with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define the minimal classes needed to install and run tigramite
INSTALL_REQUIRES = ["numpy", "scipy", "six"]
# Define all the possible extras needed
EXTRAS_REQUIRE = {
    "all": [
        "scikit-learn>=0.21",  # Gaussian Process (GP) Regression
        "matplotlib>=3.4.0",     # plotting
        "networkx>=2.4",       # plotting
        "torch>=1.7",          # GPDC torch version
        "gpytorch>=1.4",       # GPDC gpytorch version
        "dcor>=0.5.3",         # GPDC distance correlation version
    ]
}

# Define the packages needed for testing
TESTS_REQUIRE = ["nose", "pytest", "networkx>=2.4", "scikit-learn>=0.21", 
                 "torch>=1.7", "gpytorch>=1.4", "dcor>=0.5.3"]
EXTRAS_REQUIRE["test"] = TESTS_REQUIRE
# Define the extras needed for development
EXTRAS_REQUIRE["dev"] = EXTRAS_REQUIRE["all"] + TESTS_REQUIRE + ["cython"]

# Use a custom build to handle numpy.include_dirs() when building
CMDCLASS = {"build_ext": UseNumpyHeadersBuildExt}
# Define the external modules to build
EXT_MODULES = []
EXT_MODULES += define_extension("tigramite.tigramite_cython_code")

# Run the setup
setup(
    name="tigramite",
    version="4.2.2.1",
    packages=["tigramite", "tigramite.independence_tests"],
    license="GNU General Public License v3.0",
    description="Tigramite causal discovery for time series",
    author="Jakob Runge",
    author_email="jakob@jakob-runge.com",
    url="https://github.com/jakobrunge/tigramite/",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="causal inference, causal discovery, prediction, time series",
    cmdclass=CMDCLASS,
    ext_modules=EXT_MODULES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    test_suite="tests",
    tests_require=TESTS_REQUIRE,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License "
        ":: OSI Approved "
        ":: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python",
    ],
)

