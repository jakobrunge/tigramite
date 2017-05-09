# RCIT
This is an R package implementing the Randomized Conditional Independence Test (RCIT) and the Randomized conditional Correlation Test (RCoT).

# Update

I have added an additional method for estimating the null distribution called "chi2" which normalizes the partial cross-covariance statistic so that is asymptoticall follows a Gaussian with diagonal covariance. The resultant statistic therefore asymptotically obeys a central chi-squared distribution with d-degrees of freedom rather than a weighted sum of chi-squares. I find that the p-values are slightly less accurate than the default method in general. On the other hand, the new method outputs a normalized statistic, so the statistics are comparable without the same random seed. This method may be useful in algorithms like HITON-PC which initially sort variables according to the test statistic.

# Installation

The package depends on the MASS and momentchi2 packages on CRAN, so please install these first. Then:

> library(devtools)

> install_github("ericstrobl/RCIT")

> library(RCIT)

> RCIT(rnorm(1000),rnorm(1000),rnorm(1000))

