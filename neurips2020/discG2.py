import numpy as np
from scipy.stats import chi2
from scipy.special import xlogy
from tigramite.independence_tests import CondIndTest

class DiscG2(CondIndTest):
    
    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure
    
    def __init__(self,
                 **kwargs):
        
        # Specification of test
        self._measure = 'DiscG2'
        
        # Set general properties
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False
        CondIndTest.__init__(self, **kwargs)
        
    def get_dependence_measure(self, array, xyz):
        """Returns G2 test statistic"""

        # Determine the rows that correspond to the variables X, Y, and the conditions Z
        #print("Array shape:", array.shape)
        n_vars, T = array.shape
        var_index_X = [i for i in range(n_vars) if xyz[i] == 0]
        var_index_Y = [i for i in range(n_vars) if xyz[i] == 1]
        var_indices_Z = [i for i in range(n_vars) if xyz[i] == 2]

        # Determine the unique values collectively taken by the conditions Z and remember
        # which columns of 'array' correspond to each of these unique values
        uniques_Z =  {}
        for sample_index, sample in enumerate(np.transpose(array)):

            sample_Z_only = tuple(sample[tuple([var_indices_Z])])

            if sample_Z_only in uniques_Z.keys():
                uniques_Z[sample_Z_only].append(sample_index)
            else:
                uniques_Z[sample_Z_only] = [sample_index]

        #######################################################################################
        # Run through each of the unique values assumed by Z and sum up the G2 test statistic
        # and the degrees of freedom obtained from the corresponding subset of samples

        # Variables test statistic and degrees of freedom
        G2, dof = 0, 0

        # Run through all subsets (corresponding to the unique values of Z) of the samples
        for sample_indices in uniques_Z.values():

            # Restrict to samples with the same value of Z
            restricted_array = array[:, sample_indices]

            # Determine the unique values assumed by X and Y in this subset
            uniques_X = np.unique(restricted_array[var_index_X, :])
            uniques_Y = np.unique(restricted_array[var_index_Y, :])
            n_uniques_X = len(uniques_X)
            n_uniques_Y = len(uniques_Y)

            # Build a function that maps a value (x, y) of (X, Y) to its index the contingency
            # table
            x_to_cont_idx_X = {x: cont_idx_X for (cont_idx_X, x) in enumerate(uniques_X)}
            y_to_cont_idx_Y = {y: cont_idx_Y for (cont_idx_Y, y) in enumerate(uniques_Y)}

            _xy_to_cont_idx = lambda x, y: (x_to_cont_idx_X[x], y_to_cont_idx_Y[y])

            # Make the contingency table (here: s_xy) of X and Y in this subset of samples
            # as well as its marginal counts
            s_xy = np.zeros((n_uniques_X, n_uniques_Y))
            s_x = np.zeros((n_uniques_X, 1))
            s_y = np.zeros((1, n_uniques_Y))
            s = np.zeros((1, 1))
            for sample in np.transpose(restricted_array):
                x_idx, y_idx = _xy_to_cont_idx(sample[var_index_X][0], sample[var_index_Y][0])
                s_xy[x_idx, y_idx] += 1
                s_x[x_idx, 0] += 1
                s_y[0, y_idx] += 1
                s[0, 0] += 1

            # Degrees of freedom for this subset of samples
            dof_add = (n_uniques_X - 1)*(n_uniques_Y - 1)

            if dof_add > 0:
                
                # Add the G2 test statistic value for this subset of samples
                G2_subset = np.sum(2*xlogy(s_xy, s_xy*s) - 2*xlogy(s_xy, s_x*s_y))
                G2 += G2_subset

                # Add the degrees of freedom for this subset of samples
                dof += dof_add

        #######################################################################################
        
        # Write the degrees of freedom to a (temporary) instance attribute in order to pass it
        # to the signifiance functions
        self._temp_dof = dof

        # Return the test statistic
        return G2
                      
    def get_analytic_significance(self, value, T, dim):
        """Return the p_value of test statistic value 'value', according to a chi-square
        distribution with 'self._temp_dof' degrees of freedom"""
                      
        # Calculate the p_value and delete the temporary instance attribute containing
        # the degrees of freedom, which was passed from self.get_dependence_measure
        p_value = chi2.sf(value, self._temp_dof)
        del self._temp_dof
                      
        # Return p_value
        return p_value