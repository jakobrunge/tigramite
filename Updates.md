# 6/10/17
Added a permutation option to estimate the null (approx="perm"). Separated RCIT and RCoT into two separate functions.

# 5/8/17
Added an additional method for estimating the null distribution called "chi2" which normalizes the empirical partial cross-covariance matrix so that it asymptotically follows a Gaussian with diagonal covariance (after root-n multiplication). The resultant statistic therefore asymptotically obeys a central chi-squared distribution with d-degrees of freedom rather than a weighted sum of chi-squares. I find that the p-values are slightly less accurate than the default method in general. On the other hand, the new method outputs a normalized statistic, so the statistics are comparable without the same random seed. This method may be useful in algorithms like HITON-PC which initially sort variables according to the test statistic.
