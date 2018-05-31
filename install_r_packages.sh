# Install rpy2 and some dependencies
conda install rpy2
# Install r-MASS package, which is a dependency of RCIT/RCOT
R -e 'install.packages("MASS", repos = "https://cloud.r-project.org/")' 
# Install momentchi2, whichs is a dependency of RCIT/RCOT
R -e 'install.packages("momentchi2", repos = "https://cloud.r-project.org/")' 
# Install devtools so we can install from github
R -e 'install.packages("devtools", repos = "https://cloud.r-project.org/")' 
# Install RCOT/RCIT
R -e 'library(devtools); devtools::install_github("ericstrobl/RCIT")'
