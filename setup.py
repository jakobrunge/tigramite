"""
Install tigramite
"""
from __future__ import print_function
import pathlib
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import json

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

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define the minimal classes needed to install and run tigramite
INSTALL_REQUIRES =  ["numpy>=1.18", "scipy>=1.10.0", "six"]
# Define all the possible extras needed
EXTRAS_REQUIRE = {
    "all": [
        "scikit-learn>=1.2",  # Gaussian Process (GP) Regression
        "matplotlib>=3.7.0",  # plotting
        "seaborn>=0.12.2",    # plotting
        "networkx>=3.0",      # plotting
        "pytorch>=1.13.1",    # GPDC torch version
        "gpytorch>=1.9.1",    # GPDC gpytorch version
        "dcor>=0.6",          # GPDC distance correlation version
        "joblib>=1.2.0",      # CMIsymb shuffle parallelization and others
        "ortools>=9.2",       # RPCMCI
        "numba>=0.58",        # CMIknn and CMIsymb and derived classes
    ]
}

# Define the packages needed for testing
TESTS_REQUIRE = ["nose", "pytest", "networkx>=3.0", "scikit-learn>=1.2",
                 "pytorch>=1.13.1", "gpytorch>=1.9.1", "dcor>=0.6"]
EXTRAS_REQUIRE["test"] = TESTS_REQUIRE
# Define the extras needed for development
EXTRAS_REQUIRE["dev"] = EXTRAS_REQUIRE["all"]

# Use a custom build to handle numpy.include_dirs() when building
CMDCLASS = {"build_ext": UseNumpyHeadersBuildExt}

# Run the setup
setup(
    name="tigramite",
    version="5.2.6.5",
    packages=["tigramite", "tigramite.independence_tests", "tigramite.toymodels"],
    license="GNU General Public License v3.0",
    description="Tigramite causal inference for time series",
    author="Jakob Runge",
    author_email="jakob@jakob-runge.com",
    url="https://github.com/jakobrunge/tigramite/",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="causal inference, causal discovery, prediction, time series",
    cmdclass=CMDCLASS,
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

