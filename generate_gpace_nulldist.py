"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakobrunge@posteo.de>
#
# License: GNU General Public License v3.0

import numpy
import os, sys

try:
    import rpy2
    import rpy2.robjects
    rpy2.robjects.r['options'](warn=-1)

    from rpy2.robjects.packages import importr
    acepack = importr('acepack')
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
except:
    print("Couldn't import acepack, acemaxcorr estimator won't work")

try: 
    import ace
except:
    print("Could not import python ACE package for GPACE")

def generate(null_dist_samples=1000, 
    values_T=None,
    file_name = 'gpace_nulldists.npz',
    ace_version='acepack'):

    """Generates the null distribution of the ACE independence test statistic.
    
    This script generates the nulldistribution of the ACE test statistic
    under the assumption of independent uniformly distributed variables x
    and y for different sample sizes T. The values are sorted and stored as 
    a npz-file which can be loaded within tigramite_independence_tests.py

    Keyword Arguments:
        null_dist_samples {number} -- Number of samples, should be high enough 
        to reliably estimate p-values (default: {1000})
        values_T {[type]} -- Range of time series lengths / sample sizes
         (default: {None})
        file_name {str} -- File name (default: {'gpace_nulldists.npz'})
        ace_version {bool} -- Whether to use the ACE estimator implemented
        in python (pip install ace) or in R available via rpy2. The R version
        is much faster, but since GP regression mostly is the main bootleneck
        this might not matter.
    """

    if values_T is None:
        values_T = range(50, 251, 1) #+  \
               # range(250, 1001, 3) #+\
               # range(1000, 3001, 10) +\
               # range(3000, 10001, 50) +\
               # range(10000, 50001, 200)

    exact_dist = numpy.zeros((len(values_T), null_dist_samples))

    for iT, T in enumerate(values_T):

        print("Generating null distristbution for T = %d" % T)

        for i in range(null_dist_samples):

            x, y = numpy.random.rand(2, T)

            if ace_version == 'python':
                myace = ace.ace.ACESolver()
                myace.specify_data_set([x], y)
                class Suppressor(object):
                    """Wrapper class to prevent output from ACESolver."""
                    def __enter__(self):
                        self.stdout = sys.stdout
                        sys.stdout = self
                    def __exit__(self, type, value, traceback):
                        sys.stdout = self.stdout
                    def write(self, x): 
                        pass
                myace = ace.ace.ACESolver()
                myace.specify_data_set([x], y)
                with Suppressor():
                    myace.solve()
                val = numpy.corrcoef(myace.x_transforms[0],
                                     myace.y_transform)[0,1]
            else:
                ace_rpy = rpy2.robjects.r['ace'](x, y)
                val = numpy.corrcoef(numpy.asarray(ace_rpy[8]).flatten(), 
                                     numpy.asarray(ace_rpy[9]))[0, 1]

            exact_dist[iT, i] = val

        exact_dist[iT].sort()

    numpy.savez(file_name, exact_dist=exact_dist, T=numpy.array(values_T))


if __name__ == '__main__':

    # Generate null dist for python_version of ACE
    generate(null_dist_samples=1000, 
            values_T=[245],
            file_name = 'gpace_nulldists_acepack.npz',
            ace_version='acepack')