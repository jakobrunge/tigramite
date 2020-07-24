#!/bin/bash

# Steps to move from github to pip
# (Make sure pypi token exists)

# Make sure to be in a clean anaconda
# Pull tigramite github
# Checkout master

# Run all those in Linux or Windows

# Packages needed for setup
pip install --upgrade setuptools wheel twine auditwheel

# Install the package itself in the environment
pip install -e .['dev']

# Rebuild the .c files from the .pyc files
python setup.py develop

# Build the distribution
python setup.py sdist bdist_wheel


# Linux:
auditwheel repair --plat manylinux2014_x86_64 ./dist/*linux_x86_64.whl -w ./dist/
# Upload the distro
twine upload dist/*manylinux*.whl dist/*tar.gz

# Windows (EDIT names to upload all things .whl and tat.gz in dist/ !!!):
# twine upload dist/*manylinux*.whl dist/*tar.gz
