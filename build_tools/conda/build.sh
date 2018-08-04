# Install r-MASS package, which is a dependency of RCIT/RCOT
R -e "install.packages('MASS', '$PREFIX/lib/R/library', repos='https://cloud.r-project.org/')"
# Install momentchi2, whichs is a dependency of RCIT/RCOT
R -e "install.packages('momentchi2', '$PREFIX/lib/R/library', repos = 'https://cloud.r-project.org/')"
# Install the RCIT package
R -e "install.packages('external_packages/RCIT', '$PREFIX/lib/R/library', repos=NULL, type='source')"
# Install tigramite
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
echo EWEN DEBUG
echo $PREFIX
ls $PREFIX
pwd
ls
