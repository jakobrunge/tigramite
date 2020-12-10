import os, time, sys

import numpy as np
import scipy, math

import pickle 

from matplotlib import pyplot

# Imports from tigramite package available on https://github.com/jakobrunge/tigramite
import tigramite
import tigramite.data_processing as pp
from tigramite.independence_tests import ParCorr, GPDC, CMIknn

# Imports from code inside directory
import generate_data_mod as mod
import utilities as utilities
from utilities import OracleCI
import metrics_mod
from lpcmci import LPCMCI
from svarfci import SVARFCI
from svarrfci import SVARRFCI
from discG2 import DiscG2
from simulate_discrete_scm import binomial_scp, discretized_scp


# Directory to save results
folder_name = "results/"

# Arguments passed via command line
arg = sys.argv
samples = int(arg[1])
verbosity = int(arg[2])
config_list = list(arg)[3:]
num_configs = len(config_list)

time_start = time.time()

if verbosity > 1:
    plot_data = True
else:
    plot_data = False

def calculate(para_setup):

    para_setup_string, sam = para_setup

    paras = para_setup_string.split('-')
    paras = [w.replace("'","") for w in paras]

    model = str(paras[0])
    N = int(paras[1])
    n_links = int(paras[2])
    min_coeff = float(paras[3])
    coeff = float(paras[4])
    auto = float(paras[5])
    contemp_fraction = float(paras[6])
    frac_unobserved = float(paras[7])
    max_true_lag = int(paras[8])
    T = int(paras[9])

    ci_test = str(paras[10])
    method = str(paras[11])    
    pc_alpha = float(paras[12])
    tau_max = int(paras[13])

    #############################################
    ##  Data
    #############################################

        
    def lin_f(x): return x
    def f2(x): return (x + 5. * x**2 * np.exp(-x**2 / 20.))

    if model == 'autobidirected':
        if verbosity > 999:
            model_seed = verbosity - 1000
        else:
            model_seed = sam

        random_state = np.random.RandomState(model_seed)

        links ={
                0: [((0, -1), auto, lin_f), ((1, -1), coeff, lin_f)],
                1: [],
                2: [((2, -1), auto, lin_f), ((1, -1), coeff, lin_f)],                                
                3: [((3, -1), auto, lin_f), ((2, -1), min_coeff, lin_f)],                                
                }
        observed_vars = [0, 2, 3]

        noises = [random_state.randn for j in range(len(links))]

        data_all, nonstationary = mod.generate_nonlinear_contemp_timeseries(
            links=links, T=T, noises=noises, random_state=random_state)
        data = data_all[:,observed_vars]

    elif 'random' in model:
        if 'lineargaussian' in model:

            coupling_funcs = [lin_f]

            noise_types = ['gaussian'] #, 'weibull', 'uniform']
            noise_sigma = (0.5, 2)

        elif 'nonlinearmixed' in model:

            coupling_funcs = [lin_f, f2]

            noise_types = ['gaussian', 'gaussian', 'weibull']
            noise_sigma = (0.5, 2)

        if coeff < min_coeff:
            min_coeff = coeff
        couplings = list(np.arange(min_coeff, coeff+0.1, 0.1))
        couplings += [-c for c in couplings]

        auto_deps = list(np.arange(max(0., auto-0.3), auto+0.01, 0.05))

        # Models may be non-stationary. Hence, we iterate over a number of seeds
        # to find a stationary one regarding network topology, noises, etc
        if verbosity > 999:
            model_seed = verbosity - 1000
        else:
            model_seed = sam
        
        for ir in range(1000):
            # np.random.seed(model_seed)
            random_state = np.random.RandomState(model_seed)

            N_all = math.floor((N/(1.-frac_unobserved)))
            n_links_all = math.ceil(n_links/N * N_all)
            observed_vars = np.sort(random_state.choice(range(N_all), 
                size=math.ceil((1.-frac_unobserved)*N_all), replace=False)).tolist()
            
            links = mod.generate_random_contemp_model(
                N=N_all, L=n_links_all,   
                coupling_coeffs=couplings,   
                coupling_funcs=coupling_funcs,   
                auto_coeffs=auto_deps,   
                tau_max=max_true_lag,   
                contemp_fraction=contemp_fraction,  
                # num_trials=1000,  
                random_state=random_state)

            class noise_model:
                def __init__(self, sigma=1):
                    self.sigma = sigma
                def gaussian(self, T):
                    # Get zero-mean unit variance gaussian distribution
                    return self.sigma*random_state.randn(T)
                def weibull(self, T): 
                    # Get zero-mean sigma variance weibull distribution
                    a = 2
                    mean = scipy.special.gamma(1./a + 1)
                    variance = scipy.special.gamma(2./a + 1) - scipy.special.gamma(1./a + 1)**2
                    return self.sigma*(random_state.weibull(a=a, size=T) - mean)/np.sqrt(variance)
                def uniform(self, T): 
                    # Get zero-mean sigma variance uniform distribution
                    mean = 0.5
                    variance = 1./12.
                    return self.sigma*(random_state.uniform(size=T) - mean)/np.sqrt(variance)

            noises = []
            for j in links:
                noise_type = random_state.choice(noise_types)
                sigma = noise_sigma[0] + (noise_sigma[1]-noise_sigma[0])*random_state.rand()
                noises.append(getattr(noise_model(sigma), noise_type))

            if 'discretebinom' in model:
                if 'binom2' in model:
                    n_binom = 2
                elif 'binom4' in model:
                    n_binom = 4   

                data_all_check, nonstationary = discretized_scp(links=links, T = T+10000, 
                                n_binom = n_binom, random_state = random_state)
            else:
                data_all_check, nonstationary = mod.generate_nonlinear_contemp_timeseries(
                    links=links, T=T+10000, noises=noises, random_state=random_state)

            # If the model is stationary, break the loop
            if not nonstationary:
                data_all = data_all_check[:T]
                data = data_all[:,observed_vars]
                break
            else:
                print("Trial %d: Not a stationary model" % ir)
                model_seed += 10000
    else:
        raise ValueError("model %s not known"%model)

    if nonstationary:
        raise ValueError("No stationary model found: %s" % model)

    true_graph = utilities._get_pag_from_dag(links, observed_vars=observed_vars, 
        tau_max=tau_max, verbosity=verbosity)[1]

    if verbosity > 0:
        print("True Links")
        for j in links:
            print (j, links[j])
        print("observed_vars = ", observed_vars)
        print("True PAG")
        if tau_max > 0:
            for lag in range(tau_max+1):
                print(true_graph[:,:,lag])
        else:
            print(true_graph.squeeze())

    if plot_data:
        print("PLOTTING")
        for j in range(N):
            # ax = fig.add_subplot(N,1,j+1)
            pyplot.plot(data[:, j])

        pyplot.show()

    computation_time_start = time.time()

    dataframe = pp.DataFrame(data)

    #############################################
    ##  Methods
    #############################################

    # Specify conditional independence test object
    if ci_test == 'par_corr':
        cond_ind_test = ParCorr(
            significance='analytic', 
            recycle_residuals=True)
    elif ci_test == 'cmi_knn':
        cond_ind_test = CMIknn(knn=0.1, 
            sig_samples=500,
            sig_blocklength=1)
    elif ci_test == 'gp_dc':             
        cond_ind_test = GPDC(
            recycle_residuals=True)
    elif ci_test == 'discg2':
        cond_ind_test = DiscG2()
    else:
        raise ValueError("CI test not recognized.")

    if 'lpcmci' in method:
        method_paras = method.split('_')
        n_preliminary_iterations = int(method_paras[1][7:]) 

        if 'prelimonly' in method: prelim_only = True
        else: prelim_only = False

        lpcmci = LPCMCI(
            dataframe=dataframe, 
            cond_ind_test=cond_ind_test)        

        lpcmcires = lpcmci.run_lpcmci( 
                    tau_max = tau_max, 
                    pc_alpha = pc_alpha,
                    max_p_non_ancestral = 3,
                    n_preliminary_iterations = n_preliminary_iterations,
                    prelim_only = prelim_only,
                    verbosity = verbosity)
        
        graph = lpcmci.graph
        val_min = lpcmci.val_min_matrix
        max_cardinality = lpcmci.cardinality_matrix

    elif method == 'svarfci':
        svarfci = SVARFCI(
            dataframe=dataframe, 
            cond_ind_test=cond_ind_test)        
        svarfcires = svarfci.run_svarfci( 
                    tau_max = tau_max, 
                    pc_alpha = pc_alpha,
                    max_cond_px = 0,
                    max_p_dsep = 3,
                    fix_all_edges_before_final_orientation = True,
                    verbosity = verbosity)
        
        graph = svarfci.graph
        val_min = svarfci.val_min_matrix
        max_cardinality = svarfci.cardinality_matrix

    elif method == 'svarrfci':
        svarrfci = SVARRFCI(
            dataframe=dataframe, 
            cond_ind_test=cond_ind_test)        

        svarrfcires = svarrfci.run_svarrfci( 
                    tau_max = tau_max, 
                    pc_alpha = pc_alpha,
                    fix_all_edges_before_final_orientation = True,
                    verbosity = verbosity)
        
        graph = svarrfci.graph
        val_min = svarrfci.val_min_matrix
        max_cardinality = svarrfci.cardinality_matrix
    else:
        raise ValueError("%s not implemented." % method)


    computation_time_end = time.time()
    computation_time = computation_time_end - computation_time_start

    return {
            'true_graph':true_graph,
            'val_min':val_min,
            'max_cardinality':max_cardinality,

            # Method results
            'computation_time': computation_time,
            'graph':graph,
            }

if __name__ == '__main__':

    all_configs = dict([(conf, {'results':{}, 
        "graphs":{}, 
        "val_min":{}, 
        "max_cardinality":{}, 

        "true_graph":{}, 
        "computation_time":{},} ) for conf in config_list])

    job_list = [(conf, i) for i in range(samples) for conf in config_list]

    num_tasks = len(job_list)

    for config_sam in job_list:
        config, sample = config_sam
        print("Experiment %s - Realization %d" %(config, sample))
        all_configs[config]['results'][sample] = calculate(config_sam)

    print("\nsaving all configs...")

    for conf in list(all_configs.keys()):
        all_configs[conf]['graphs'] = np.zeros((samples, ) + all_configs[conf]['results'][0]['graph'].shape, dtype='<U3')
        all_configs[conf]['true_graphs'] = np.zeros((samples, ) + all_configs[conf]['results'][0]['true_graph'].shape, dtype='<U3')
        all_configs[conf]['val_min'] = np.zeros((samples, ) + all_configs[conf]['results'][0]['val_min'].shape)
        all_configs[conf]['max_cardinality'] = np.zeros((samples, ) + all_configs[conf]['results'][0]['max_cardinality'].shape)
        all_configs[conf]['computation_time'] = [] 

        for i in list(all_configs[conf]['results'].keys()):
            all_configs[conf]['graphs'][i] = all_configs[conf]['results'][i]['graph']
            all_configs[conf]['true_graphs'][i] = all_configs[conf]['results'][i]['true_graph']
            all_configs[conf]['val_min'][i] = all_configs[conf]['results'][i]['val_min']
            all_configs[conf]['max_cardinality'][i] = all_configs[conf]['results'][i]['max_cardinality']

            all_configs[conf]['computation_time'].append(all_configs[conf]['results'][i]['computation_time'])
    
        # Save all results
        file_name = folder_name + '%s' %(conf)

        # Compute and save metrics in separate (smaller) file
        metrics = metrics_mod.get_evaluation(results=all_configs[conf])
        for metric in metrics:
            if metric != 'computation_time':
                print(f"{metric:30s} {metrics[metric][0]: 1.2f} +/-{metrics[metric][1]: 1.2f} ")
            else:
                print(f"{metric:30s} {metrics[metric][0]: 1.2f} +/-[{metrics[metric][1][0]: 1.2f}, {metrics[metric][1][1]: 1.2f}]")

        print("Metrics dump ", file_name.replace("'", "").replace('"', '') + '_metrics.dat')
        file = open(file_name.replace("'", "").replace('"', '') + '_metrics.dat', 'wb')
        pickle.dump(metrics, file, protocol=-1)        
        file.close()

        del all_configs[conf]['results']

        # Also save raw results
        print("dump ", file_name.replace("'", "").replace('"', '') + '.dat')
        file = open(file_name.replace("'", "").replace('"', '') + '.dat', 'wb')
        pickle.dump(all_configs[conf], file, protocol=-1)        
        file.close()


    time_end = time.time()
    print('Run time in hours ', (time_end - time_start)/3600.)