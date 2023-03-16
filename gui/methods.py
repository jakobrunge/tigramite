"""This file contains all the tigramite functionality"""
# License: GNU General Public License v3.0
import numpy as np
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.parcorr_mult import ParCorrMult
from tigramite.independence_tests.parcorr_wls import ParCorrWLS
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.gpdc_torch import GPDCtorch
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmiknnmixed import CMIknnMixed
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.regressionCI import RegressionCI
from tigramite.pcmci import PCMCI


def calculate_results(Data, Test, Method, test_params, method_params, terminal_out):
    with terminal_out:
        try:
            data = np.load(Data)
            dataframe = pp.DataFrame(data)
            cond_ind_test = get_cond_ind_test(Test, test_params)
            pcmci = PCMCI(
                dataframe=dataframe,
                cond_ind_test=cond_ind_test,
                verbosity=1)
            if method_params:
                if Method == "PCMCI":
                    results = pcmci.run_pcmci(**method_params)
                elif Method == "PCMCI+":
                    results = pcmci.run_pcmciplus(**method_params)
            else:
                if Method == "PCMCI":
                    results = pcmci.run_pcmci()
                elif Method == "PCMCI+":
                    results = pcmci.run_pcmciplus()
            return pcmci, results
        except Exception as e:
            print(str(e))
    return {}, {}



def make_plot(plot_type, pcmci, results, plot_out, alpha_value, plot_parameters):
    """Makes causal graphs from results, uses tigramite functionality and displays it in the PlotOut plot_out"""
    with plot_out:
        try:
            #q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
            #graph = pcmci.get_graph_from_pmatrix(p_matrix=q_matrix, alpha_level=0.01, tau_min=0, tau_max=1, link_assumptions=None)
            #results['graph'] = graph
            if plot_type == "Process Graph":
                tp.plot_graph(val_matrix=results['val_matrix'],
                              graph=results['graph'],
                              **plot_parameters
                              )
            elif plot_type == "Time Series Graph":
                tp.plot_time_series_graph(
                    val_matrix=results['val_matrix'],
                    graph=results['graph'],
                    **plot_parameters
                    )
            elif plot_type == "Lagged Correlation":
                correlations = pcmci.get_lagged_dependencies(tau_max=20, val_only=True)['val_matrix']
                lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names': var_names,
                                                                                        'x_base': 5, 'y_base': .5})
            else:
                print("Choose one of the graphs from the drop down menu")
            plt.show()
        except Exception as e:
            print(str(e))
        return True


def get_cond_ind_test(test, test_params):
    """Selects conditional independence test and returns it"""
    if test_params:
        if test == "ParCorr":
            return ParCorr(**test_params)
        elif test == "RobustParCorr":
            return RobustParCorr(**test_params)
        elif test == "ParCorrMult":
            return ParCorrMult(**test_params)
        elif test == "ParCorrWLS":
            return ParCorrWLS(**test_params)
        elif test == "GPDC":
            return GPDC(**test_params)
        elif test == "GPDCtorch":
            return GPDCtorch(**test_params)
        elif test == "CMIknn":
            return CMIknn(**test_params)
        elif test == "CMIsymb":
            return CMIsymb(**test_params)
        elif test == "Gsquared":
            return Gsquared(**test_params)
        elif test == "CMIknnMixed":
            return CMIknnMixed(**test_params)
        elif test == "RegressionCI":
            return RegressionCI(**test_params)
        else:
            raise Exception("Unknown test or parameters")
    else:
        if test == "ParCorr":
            return ParCorr()
        elif test == "RobustParCorr":
            return RobustParCorr()
        elif test == "ParCorrMult":
            return ParCorrMult()
        elif test == "ParCorrWLS":
            return ParCorrWLS()
        elif test == "GPDC":
            return GPDC()
        elif test == "GPDCtorch":
            return GPDCtorch()
        elif test == "CMIknn":
            return CMIknn()
        elif test == "CMIsymb":
            return CMIsymb()
        elif test == "Gsquared":
            return Gsquared()
        elif test == "CMIknnMixed":
            return CMIknnMixed()
        elif test == "RegressionCI":
            return RegressionCI()
        else:
            raise Exception("Unknown test or parameters")