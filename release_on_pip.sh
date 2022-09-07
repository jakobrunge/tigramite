#!/bin/bash

homename="jakobrunge"
version="51030"

# Steps to move from github to pip
# (Make sure pypi token exists)

# Change version number in setup.py
# Merge developer into Master on github

# Make sure docs are updated on master
# tigramite/docs$ make -C . html; cp -r _build/html/* .;
# tigramite$ git add docs/*

# Create folder for next version (e.g., version 4.2.1.0 as edited in setup.py)
mkdir /home/$homename/work/code/python_code/tigramite/tigramite_v4/tigramitepipdistribute/tigramitepypi/v$version
cd /home/$homename/work/code/python_code/tigramite/tigramite_v4/tigramitepipdistribute/tigramitepypi/v$version

# Make sure to be in a clean anaconda
conda create --name tigramite-release-$version python=3.9 anaconda -y
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tigramite-release-$version

# Pull tigramite github
git clone https://github.com/jakobrunge/tigramite.git
cd tigramite

# Checkout master
git checkout master

# Run all those in Linux or Windows

# Packages needed for setup (the parameter "--use-feature=2020-resolver" may not be necessary anymore soon )
pip install --upgrade setuptools wheel twine auditwheel

# Install the package itself in the environment
pip install -e .['dev']

# Rebuild the .c files from the .pyc files
# python setup.py develop

# Build the distribution
python setup.py sdist bdist_wheel


# Linux:
# auditwheel repair --plat manylinux2014_x86_64 ./dist/*linux_x86_64.whl -w ./dist/
# auditwheel repair --plat manylinux2014_x86_64 ./dist/*.whl -w ./dist/


# Upload the distro
# twine upload dist/*manylinux*.whl dist/*tar.gz
twine upload dist/*.whl dist/*tar.gz


# Windows (EDIT names to upload all things .whl and tat.gz in dist/ !!!):
# twine upload dist/*manylinux*.whl dist/*tar.gz
