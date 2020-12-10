# Code repository for paper "High-recall causal discovery for autocorrelated time series with latent confounders" accepted for publication at NeurIPS 2020

This repository hosts the code conntect our paper *High-recall causal discovery for autocorrelated time series with latent confounders* that has been accepted for publication at NeurIPS 2020. Below, we show you how to reproduce the presented results.

Currently, the paper is already available on the arXiv: https://arxiv.org/abs/2007.01884.


## Requirements

We provide the file `anaconda-packages.txt` to create an anaconda environment that includes the relevant packages. The *tigramite* package (of which this repository is part of) is publicly available at GitHub and can be downloaded from https://github.com/jakobrunge/tigramite/tree/master.

To create the anaconda environment run:

$ conda create --name \<env\> --file anaconda-packages.txt

Then activate this environment with:

$ conda activate \<env\>

To install tigramite, go into the tigramite-master directory and run:

$ python setup.py install

This should setup all required python packages.


## Implemented methods

In our simulation studies we compare three methods: SVAR-FCI (baseline), SVAR-RFCI (baseline) and LPCMCI (our proposed method). These methods are respectively provided in the files `svarfci.py`, `svarrfci.py`, and `lpcmci.py`. The methods are implemented as classes, which are then loaded by the script used for running the simulations (see below).


## Script for running simulations studies

The script `compute_experiments.py` generates results of the three methods (LPCMCI, SVAR-FCI, SVAR-RFCI) applied to different datasets. It is run as:

$ python compute_experiments.py num_realizations verbosity configuration

with command line arguments:
* num_realizations = integer denoting the number of time series realizations to generate
* verbosity = integer for verbosity in method output
* configuration = string that identifies a particular experiment consisting of a model and method. 

The configuration string specifies the different parameters used by the methods. The syntax is as follows:

'model-N-L-min_coeff-max_coeff-autocorr-frac_contemp_links-frac_unobserved-max_true_lag-time_series_length-CI_test-method-alpha_level-tau_max'

Here, the individual parameters are:
* model: model system
* N: number of variables
* L: number of links
* min_coeff: minimum coefficient
* max_coeff: max. coefficient
* autocorr: max. autocorr
* frac_contemp_links: fraction of contemporaneous links
* frac_unobserved: fraction of unobserved variables
* max_true_lag: max. true time lag
* time_series_length: length of realizations
* CI_test: conditional independence test from tigramite package
* method: method name
* alpha_level: significance level
* tau_max: maximum time lag in method

Results are saved into the folder `results`, which needs to be created before the first application of `compute_experiments.py` by the command:

$ mkdir results

Here are a few examples of running the `compute_experiments.py` script:

For SVAR-FCI run:

$ python compute_experiments.py 100 0 'random_lineargaussian-3-3-0.2-0.8-0.9-0.3-0.3-3-100-par_corr-svarfci-0.05-5'

For SVAR-RFCI run:

$ python compute_experiments.py 100 0 'random_lineargaussian-3-3-0.2-0.8-0.9-0.3-0.3-3-100-par_corr-svarrfci-0.05-5'

For LCPCMI with k=0 run:

$ python compute_experiments.py 100 0 'random_lineargaussian-3-3-0.2-0.8-0.9-0.3-0.3-3-100-par_corr-lpcmci_nprelim0-0.05-5'

For LCPCMI with k=4 run:

$ python compute_experiments.py 100 0 'random_lineargaussian-3-3-0.2-0.8-0.9-0.3-0.3-3-100-par_corr-lpcmci_nprelim4-0.05-5'

The results are saved as Python dictionaries into the `results` folder, unless you have specified a different folder in line 26 of the script. To compute several configurations in a row (here for different autocorrelations a=0., 0.5, 0.9, 0.95), run:

$ python compute_experiments.py 100 0 'random_lineargaussian-5-5-0.2-0.8-0.0-0.3-0.3-3-500-par_corr-lpcmci_nprelim4-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.5-0.3-0.3-3-500-par_corr-lpcmci_nprelim4-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.9-0.3-0.3-3-500-par_corr-lpcmci_nprelim4-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.95-0.3-0.3-3-500-par_corr-lpcmci_nprelim4-0.01-5'


## Script for plotting results

First generate the results for all desired parameter configurations as described in the previous paragraph. Then create a folder `figures` for saving the plots:

$ mkdir figures  

The script `plot_experiments.py` then uses the Python dictionaries inside the `results` folder to create figure panels as in Fig. 2 of the main paper, which it saves into the `figures` folder. (You can change these folders in lines 569 and 570 of the script, but make sure to remain consistent among both scripts.)

The script is run as

$ python plot_experiments.py ci_test variant

with the command line arguments:
* ci_test = conditional independence test from tigramite package
* variant = string that identifies a figure setup.

The following examples generate a number of figures that show the performance of the methods against different variables (x-axis in Fig. 2 main text) for different other experiment parameters such as N, T, etc. The individual figures show the particular setup in the top right, and their names also indicate the respective parameters.

* Varying autocorrelation for ParCorr CI test (Fig. 2B):
$ python plot_experiments.py par_corr autocorr

* Varying autocorrelation for GPDC CI test (Supplement):
$ python plot_experiments.py gp_dc autocorr

* Varying number of variables for ParCorr CI test (Fig. 2C):
$ python plot_experiments.py par_corr highdim

* Varying maximum time lag for ParCorr CI test (Fig. 2D):
$ python plot_experiments.py par_corr tau_max

* Varying sample size for ParCorr CI test (Supplement):
$ python plot_experiments.py par_corr sample_size

* Varying fraction of unobserved variables for ParCorr CI test (Supplement):
$ python plot_experiments.py par_corr unobserved

Other desired method comparisons (e.g. LPCMCI for different k) can be chosen in the plot script. Note that if not all corresponding results have been created before plotting, the created plots may be only partially filled or empty.


## Example: How to create Fig. 2B in the main text

To create the plot shown in Fig. 2B (upper right) of the main text, run the following commands (this may take a while):

$ python compute_experiments.py 500 0 'random_lineargaussian-5-5-0.2-0.8-0.0-0.3-0.3-3-500-par_corr-svarfci-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.5-0.3-0.3-3-500-par_corr-svarfci-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.9-0.3-0.3-3-500-par_corr-svarfci-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.95-0.3-0.3-3-500-par_corr-svarfci-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.99-0.3-0.3-3-500-par_corr-svarfci-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.0-0.3-0.3-3-500-par_corr-svarrfci-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.5-0.3-0.3-3-500-par_corr-svarrfci-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.9-0.3-0.3-3-500-par_corr-svarrfci-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.95-0.3-0.3-3-500-par_corr-svarrfci-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.99-0.3-0.3-3-500-par_corr-svarrfci-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.0-0.3-0.3-3-500-par_corr-lpcmci_nprelim0-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.5-0.3-0.3-3-500-par_corr-lpcmci_nprelim0-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.9-0.3-0.3-3-500-par_corr-lpcmci_nprelim0-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.95-0.3-0.3-3-500-par_corr-lpcmci_nprelim0-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.99-0.3-0.3-3-500-par_corr-lpcmci_nprelim0-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.0-0.3-0.3-3-500-par_corr-lpcmci_nprelim4-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.5-0.3-0.3-3-500-par_corr-lpcmci_nprelim4-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.9-0.3-0.3-3-500-par_corr-lpcmci_nprelim4-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.95-0.3-0.3-3-500-par_corr-lpcmci_nprelim4-0.01-5' 'random_lineargaussian-5-5-0.2-0.8-0.99-0.3-0.3-3-500-par_corr-lpcmci_nprelim4-0.01-5'

$ python plot_experiments.py par_corr autocorr

This will create several PDF documents, one of which is the desired plot (the runtime estimates may deviate).

## Discrete Examples

To create the results for discrete models with the discrete G2 conditional independence test, run the following commands:

$ python compute_experiments.py 500 0 'random_lineargaussian_discretebinom2-4-4-0.2-0.8-0.0-0.3-0.3-3-2000-discg2-svarfci-0.01-5'

Here "binom2" implies a model with n_{bin} = 2 as described in the paper.

## Real data example

We further provide the Jupyter notebook `river_discharge.ipynb` to reproduce the real data example that is discussed in Section 5 of the paper. *(Note: This refers to the final NeurIPS 2020 submission. The respective section is not yet included in the arXiv version.)*

To start the notebook make sure to have the anaconda environment activated and run:

$ jupyter notebook river_discharge.ipynb