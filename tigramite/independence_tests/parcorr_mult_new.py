from tigramite.independence_tests.parcorr_mult import ParCorrMult
import numpy as np
import warnings
from copy import deepcopy


# make adaption in ParCorr-Mult of checking for constant rows and not including the result
class ParCorrMultNew(ParCorrMult):
    def __init__(self, **kwargs):

        ParCorrMult.__init__(self, **kwargs)

    def get_dependence_measure(self, array, xyz):
        """Return multivariate kernel correlation coefficient.
        Estimated as some dependency measure on the
        residuals of a linear OLS regression.
        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns
        xyz : array of ints
            XYZ identifier array of shape (dim,).
        Returns
        -------
        val : float
            Partial correlation coefficient.
        """

        dim, T = array.shape
        dim_x = (xyz == 0).sum()
        dim_y = (xyz == 1).sum()

        x_vals, xyz_new = self._get_single_residuals(array, xyz, target_var=0)
        y_vals, xyz_new = self._get_single_residuals(array, xyz, target_var=1)

        dim_x = (xyz_new == 0).sum()
        dim_y = (xyz_new == 1).sum()

        array_resid = np.vstack((x_vals.reshape(dim_x, T), y_vals.reshape(dim_y, T)))
        xyz_resid = np.array([index_code for index_code in xyz_new if index_code != 2])

        val = self.mult_corr(array_resid, xyz_resid)

        return val

    def _get_single_residuals(self, array, xyz, target_var,
                              standardize=True,
                              return_means=False):
        """Returns residuals of linear multiple regression.
        Performs a OLS regression of the variable indexed by target_var on the
        conditions Z. Here array is assumed to contain X and Y as the first two
        rows with the remaining rows (if present) containing the conditions Z.
        Optionally returns the estimated regression line.
        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns
        xyz : array of ints
            XYZ identifier array of shape (dim,).
        target_var : {0, 1}
            Variable to regress out conditions from.
        standardize : bool, optional (default: True)
            Whether to standardize the array beforehand. Must be used for
            partial correlation.
        return_means : bool, optional (default: False)
            Whether to return the estimated regression line.
        Returns
        -------
        resid [, mean] : array-like
            The residual of the regression and optionally the estimated line.
        """

        array_copy = deepcopy(array)
        xyz_copy = deepcopy(xyz)
        # remove the parts of the array within dummy that are constant zero (ones are cut off)
        mask = np.all(array_copy == array_copy[:, 0, None], axis=1)
        xyz_copy = xyz_copy[~mask]
        array_copy = array_copy[~mask]

        dim, T = array_copy.shape
        dim_z = (xyz_copy == 2).sum()

        # Standardize
        if standardize:
            array_copy -= array_copy.mean(axis=1).reshape(dim, 1)
            std = array_copy.std(axis=1)
            for i in range(dim):
                if std[i] != 0.:
                    array_copy[i] /= std[i]
            if np.any(std == 0.):
                warnings.warn("Possibly constant array!")
            # array /= array.std(axis=1).reshape(dim, 1)
            # if np.isnan(array).sum() != 0:
            #     raise ValueError("nans after standardizing, "
            #                      "possibly constant array!")

        y = np.fastCopyAndTranspose(array_copy[np.where(xyz_copy == target_var)[0], :])

        if dim_z > 0:
            z = np.fastCopyAndTranspose(array_copy[np.where(xyz_copy == 2)[0], :])
            beta_hat = np.linalg.lstsq(z, y, rcond=None)[0]
            mean = np.dot(z, beta_hat)
            resid = y - mean
        else:
            resid = y
            mean = None

        if return_means:
            return (np.fastCopyAndTranspose(resid), np.fastCopyAndTranspose(mean))

        return np.fastCopyAndTranspose(resid), xyz_copy

    def mult_corr(self, array, xyz, standardize=True):
        """Return multivariate dependency measure.

        Parameters
        ----------
        array : array-like
            data array with X, Y in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        standardize : bool, optional (default: True)
            Whether to standardize the array beforehand. Must be used for
            partial correlation.

        Returns
        -------
        val : float
            Multivariate dependency measure.
        """
        array_copy = deepcopy(array)
        xyz_copy = deepcopy(xyz)
        # remove the parts of the array within dummy that are constant zero (ones are cut off)
        mask = np.all(array_copy == array_copy[:, 0, None], axis=1)
        xyz_copy = xyz_copy[~mask]
        array_copy = array_copy[~mask]

        dim, n = array_copy.shape
        dim_x = (xyz_copy == 0).sum()
        dim_y = (xyz_copy == 1).sum()

        # Standardize
        if standardize:
            array_copy -= array_copy.mean(axis=1).reshape(dim, 1)
            std = array_copy.std(axis=1)
            for i in range(dim):
                if std[i] != 0.:
                    array_copy[i] /= std[i]
            if np.any(std == 0.):
                warnings.warn("Possibly constant array!")
                print(array_copy)
            # array /= array.std(axis=1).reshape(dim, 1)
        if np.isnan(array).sum() != 0:
            raise ValueError("nans after standardizing, "
                             "possibly constant array!")

        x = array_copy[np.where(xyz_copy == 0)[0]]
        y = array_copy[np.where(xyz_copy == 1)[0]]

        if self.correlation_type == 'max_corr':
            # Get (positive or negative) absolute maximum correlation value
            corr = np.corrcoef(x, y)[:len(x), len(x):].flatten()
            if corr.size > 0:
                val = corr[np.argmax(np.abs(corr))]
            else:
                val = 0.
                print("EMPTY CORR")

            # val = 0.
            # for x_vals in x:
            #     for y_vals in y:
            #         val_here, _ = stats.pearsonr(x_vals, y_vals)
            #         val = max(val, np.abs(val_here))

        # elif self.correlation_type == 'linear_hsci':
        #     # For linear kernel and standardized data (centered and divided by std)
        #     # biased V -statistic of HSIC reduces to sum of squared inner products
        #     # over all dimensions
        #     val = ((x.dot(y.T)/float(n))**2).sum()
        else:
            raise NotImplementedError("Currently only"
                                      "correlation_type == 'max_corr' implemented.")

        return val