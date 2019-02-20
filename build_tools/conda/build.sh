# NOTE: when installing, conda will include all things under $PREFIX as a part 
# of your distributed package.  As such, the R libraries are installed to 
# the correct library location under $PREFIX. This may not be the most robust 
# solution, but it works for now.

R_LIBS_BUILD=$PREFIX/lib/R/library
# Install r-MASS package, which is a dependency of RCIT/RCOT
R -e "install.packages('MASS', '$R_LIBS_BUILD', repos='https://cloud.r-project.org/')"
# Install momentchi2, whichs is a dependency of RCIT/RCOT
R -e "install.packages('momentchi2', '$R_LIBS_BUILD', repos = 'https://cloud.r-project.org/')"
# Install the RCIT package
R -e "install.packages('external_packages/RCIT', '$R_LIBS_BUILD', repos=NULL, type='source')"
# Install tigramite
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
