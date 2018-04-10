# RCIT and RCoT
This is an R package implementing the Randomized Conditional Independence Test (RCIT) and the Randomized conditional Correlation Test (RCoT). We recommend using RCoT as the default conditional independence test, since it is more accurate than RCIT.

# Installation

The package depends on the MASS and momentchi2 packages on CRAN, so please install these first. Then:

> library(devtools)

> install_github("ericstrobl/RCIT")

> library(RCIT)

> RCIT(rnorm(1000),rnorm(1000),rnorm(1000))

> RCoT(rnorm(1000),rnorm(1000),rnorm(1000))

