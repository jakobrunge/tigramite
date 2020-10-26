import matplotlib as mpl
# print([key for key in list(mpl.rcParams.keys()) if 'pad' in key])
params = { 'figure.figsize': (8, 10),
           'legend.fontsize': 8,
           # 'title.fontsize': 8,
           'lines.color':'black',
           'lines.linewidth':1,
            'xtick.labelsize':4,
            'xtick.major.pad'  : 3, 
            'xtick.major.size' : 2,
            'ytick.major.pad'  : 3,
            'ytick.major.size' : 2,
            'ytick.labelsize':7,
            'axes.labelsize':8,
            'font.size':8,
            'axes.labelpad':2,
            # 'text.usetex' : True,
            # 'legend.labelsep': 0.0005 
            }
import collections
from matplotlib.ticker import ScalarFormatter, NullFormatter
from matplotlib import gridspec
import matplotlib.cm as cm

mpl.rcParams.update(params)


import sys, os
import numpy as np
import pylab
import matplotlib.pyplot as plt
import pickle
import scipy

import matplotlib.pyplot as plt

arg = sys.argv
ci_test = str(arg[1])  
variant = str(arg[2]) 

def method_label(method):
    # return method
    # if not 'paper' in variant:
    #     return method

    if method == 'svarfci':
        return 'SVAR-FCI'
    elif method == 'svarrfci':
        return 'SVAR-RFCI'   
    elif 'lpcmci' in method:
        if 'prelimonly' in method and 'prelim1' in method:
            return r'LPCMCI$(l=0)$'
        elif 'prelim0' in method:
            return r'LPCMCI$(k=0)$'
        elif 'prelim1' in method:
            return r'LPCMCI$(k=1)$'
        elif 'prelim2' in method:
            return r'LPCMCI$(k=2)$'
        elif 'prelim3' in method:
            return r'LPCMCI$(k=3)$'
        elif 'prelim4' in method:
            return r'LPCMCI$(k=4)$'
    else:
        return method

name = {'par_corr':r'ParCorr', 'gp_dc':r'GPDC', 'cmi_knn':r'CMIknn'}


def get_metrics_from_file(para_setup):

    name_string = '%s-'*len(para_setup)  # % para_setup
    name_string = name_string[:-1]

    try:
        print("load from metrics file  %s_metrics.dat " % (folder_name + name_string % tuple(para_setup)))
        results = pickle.load(open(folder_name + name_string % tuple(para_setup) + '_metrics.dat', 'rb'), encoding='latin1')
    except:
        print('failed from metrics file '  , tuple(para_setup))
        return None

    return results


def print_time(seconds, precision=1):
    if precision == 0:
        if seconds > 60*60.:
            return "%.0fh" % (seconds/3600.)
        elif seconds > 60.:
            return "%.0fmin" % (seconds/60.)
        else:
            return "%.0fs" % (seconds)
    else:
        if seconds > 60*60.:
            return "%.1fh" % (seconds/3600.)
        elif seconds > 60.:
            return "%.1fmin" % (seconds/60.)
        else:
            return "%.1fs" % (seconds)

def print_time_std(time, precision=1):

    mean = time.mean()
    std = time.std()
    if precision == 0:
        if mean > 60*60.:
            return r"%.0f$\pm$%.0fh" % (mean/3600., std/3600.)
        elif mean > 60.:
            return r"%.0f$\pm$%.0fmin" % (mean/60., std/60.)
            # return "%.0fmin" % (mean/60.)
        else:
            return r"%.0f$\pm$%.0fs" % (mean, std)
            # return "%.0fs" % (mean)
    else:
        if mean > 60*60.:
            return r"%.1f$\pm$%.1fh" % (mean/3600., std/3600.)
        elif mean > 60.:
            return r"%.1f$\pm$%.1fmin" % (mean/60., std/60.)
            # return "%.0fmin" % (mean/60.)
        else:
            return r"%.1f$\pm$%.1fs" % (mean, std)

def draw_it(paras, which):

    figsize = (4, 2.5)  #(4, 2.5)
    capsize = .5
    marker1 = 'o'
    marker2 = 's'
    marker3 = '+'
    alpha_marker = 1.

    params = { 
           'legend.fontsize': 5,
           'legend.handletextpad': .05,
           # 'title.fontsize': 8,
           'lines.color':'black',
           'lines.linewidth':.5,
           'lines.markersize':2,
           # 'lines.capsize':4,
            'xtick.labelsize':4,
            'xtick.major.pad'  : 1, 
            'xtick.major.size' : 2,
            'ytick.major.pad'  : 1,
            'ytick.major.size' : 2,
            'ytick.labelsize':4,
            'axes.labelsize':8,
            'font.size':8,
            'axes.labelpad':2,
            # 'axes.grid': True,
            'axes.spines.right' : False,
            'axes.spines.top' : False,
            # 'lines.clip_on':False,
            # 'axes.spines.left.outward' : 4,
            # 'text.usetex' : True,
            # 'legend.labelsep': 0.0005 
            }
    mpl.rcParams.update(params)



    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4)
    ax1a = fig.add_subplot(gs[0, 0])
    ax1b = fig.add_subplot(gs[1, 0])

    ax2a = fig.add_subplot(gs[0, 1])
    ax2b = fig.add_subplot(gs[1, 1])

    ax3a = fig.add_subplot(gs[0, 2])
    ax3b = fig.add_subplot(gs[1, 2])

    # ax4 = fig.add_subplot(gs[:, 3])
    ax4a = fig.add_subplot(gs[0, 3])
    ax4b = fig.add_subplot(gs[1, 3])

    if fpr_precision == 'fpr':
        if which == 'pc_alpha':
            print(paras)
            ax1b.plot(paras, paras, color='grey', linewidth=2.)
        else:    
            ax1b.axhline(pc_alpha, color='grey', linewidth=2.)



    for method in methods:
        for para in paras:

            # para_plot = para + 0.04*np.random.rand()*abs(paras[-1]-paras[0])
            para_plot = paras.index(para) + methods.index(method)/float(len(methods))*.6

            if which == 'auto':
                auto_here = para
                N_here = N
                tau_max_here = tau_max
                frac_unobserved_here = frac_unobserved
                pc_alpha_here = pc_alpha
                T_here = T

            elif which == 'N':
                N_here = para
                auto_here = auto
                tau_max_here = tau_max
                frac_unobserved_here = frac_unobserved
                pc_alpha_here = pc_alpha
                T_here = T

            elif which == 'tau_max':
                N_here = N
                auto_here = auto
                tau_max_here = para
                frac_unobserved_here = frac_unobserved
                pc_alpha_here = pc_alpha
                T_here = T

            elif which == 'sample_size':
                N_here = N
                auto_here = auto
                tau_max_here = tau_max
                frac_unobserved_here = frac_unobserved
                pc_alpha_here = pc_alpha
                T_here = para

            elif which == 'unobserved':
                N_here = N
                auto_here = auto
                tau_max_here = tau_max
                frac_unobserved_here = para
                pc_alpha_here = pc_alpha
                T_here = T

            if N_here == 2:
                n_links_here = 1
            else:
                n_links_here = links_from_N(N_here)


            para_setup = (model, N_here, n_links_here, min_coeff, coeff, auto_here, contemp_fraction, frac_unobserved_here,  
                            max_true_lag, T_here, ci_test, method, pc_alpha_here, tau_max_here) 

            metrics_dict = get_metrics_from_file(para_setup)
            if metrics_dict is not None:

                ax1a.errorbar(para_plot, *metrics_dict['adj_lagged_recall'], capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker=marker1, linestyle='solid')
                ax1a.errorbar(para_plot, *metrics_dict['adj_auto_recall'], capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker=marker3, linestyle='solid')
                ax1a.errorbar(para_plot, *metrics_dict['adj_contemp_recall'], capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker=marker2, linestyle='dashed')

                ax1b.errorbar(para_plot, *metrics_dict['adj_lagged_%s' % fpr_precision], capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker=marker1, linestyle='solid')
                ax1b.errorbar(para_plot, *metrics_dict['adj_auto_%s' % fpr_precision], capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker=marker3, linestyle='solid')
                ax1b.errorbar(para_plot, *metrics_dict['adj_contemp_%s' % fpr_precision], capsize=capsize,  alpha=alpha_marker,
                    color=color_picker(method), marker=marker2, linestyle='dashed')

                ax2a.errorbar(para_plot, *metrics_dict['edgemarks_lagged_recall'], capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker=marker1, linestyle='solid')
                ax2a.errorbar(para_plot, *metrics_dict['edgemarks_auto_recall'], capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker=marker3, linestyle='solid')
                ax2a.errorbar(para_plot, *metrics_dict['edgemarks_contemp_recall'], capsize=capsize,  alpha=alpha_marker,
                    color=color_picker(method), marker=marker2, linestyle='dashed')

                ax2b.errorbar(para_plot, *metrics_dict['edgemarks_lagged_precision'], capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker=marker1, linestyle='solid')
                ax2b.errorbar(para_plot, *metrics_dict['edgemarks_auto_precision'], capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker=marker3, linestyle='solid')
                ax2b.errorbar(para_plot, *metrics_dict['edgemarks_contemp_precision'], capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker=marker2, linestyle='dashed')

                ax3a.errorbar(para_plot, *metrics_dict['valmin_lagged'], capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker=marker1)
                ax3a.errorbar(para_plot, *metrics_dict['valmin_auto'], capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker=marker3)
                ax3a.errorbar(para_plot, *metrics_dict['valmin_contemp'], capsize=capsize,  alpha=alpha_marker,
                    color=color_picker(method), marker=marker2)

                ax3b.errorbar(para_plot, *metrics_dict['cardinality_lagged'], capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker=marker1)
                ax3b.errorbar(para_plot, *metrics_dict['cardinality_auto'], capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker=marker3)
                ax3b.errorbar(para_plot, *metrics_dict['cardinality_contemp'], capsize=capsize,  alpha=alpha_marker,
                    color=color_picker(method), marker=marker2)

                ax4a.errorbar(para_plot, metrics_dict['computation_time'][0], metrics_dict['computation_time'][1].reshape(2, 1), capsize=capsize, alpha=alpha_marker,
                    color=color_picker(method), marker='p', linestyle='solid')

                if method == methods[0]:
                    ax4b.plot(para_plot, metrics_dict['directed_anylink'][0], alpha=alpha_marker,
                        color='black', marker='>')
                    ax4b.plot(para_plot, metrics_dict['bidirected_anylink'][0], alpha=alpha_marker,
                        color='black', marker='D')
                    unoriented = 1. -  metrics_dict['directed_anylink'][0] - metrics_dict['bidirected_anylink'][0]
                    ax4b.plot(para_plot, unoriented, alpha=alpha_marker,
                        color='black', marker='o', fillstyle='none')
 
    # print(axes)
    axes = {'ax1a':ax1a, 'ax1b':ax1b, 'ax2a':ax2a, 'ax2b':ax2b, 'ax3a':ax3a, 'ax3b':ax3b, 'ax4a':ax4a, 'ax4b':ax4b}
    for axname in axes:

        ax = axes[axname]

        if which == 'N':
            # print(ax)
            # print(axes)
            ax.set_xlim(-0.5, len(paras))
            if ci_test == 'par_corr':
                ax.xaxis.set_ticks([paras.index(p) for p in paras] )
                ax.xaxis.set_ticklabels([str(p) for p in paras] )
            else:
                ax.xaxis.set_ticks([paras.index(p) for p in paras] )
                ax.xaxis.set_ticklabels([str(p) for p in paras] )
        elif which == 'auto':
            ax.set_xlim(0, len(paras))
            ax.xaxis.set_ticks([paras.index(p) for p in paras] )
            ax.xaxis.set_ticklabels([str(p) for p in paras] )

        elif which == 'tau_max':
            ax.set_xlim(-0.5, len(paras))
            ax.xaxis.set_ticks([paras.index(p) for p in paras] )
            ax.xaxis.set_ticklabels([str(p) for p in paras] )

        elif which == 'unobserved':
            ax.set_xlim(0, len(paras))
            ax.xaxis.set_ticks([paras.index(p) for p in paras] )
            ax.xaxis.set_ticklabels([str(p) for p in paras] )

        elif which == 'sample_size':
            ax.set_xlim(0, len(paras))
            ax.xaxis.set_ticks([paras.index(p) for p in paras] )
            ax.xaxis.set_ticklabels([str(p) for p in paras] )

        # ax.set_xlabel(xlabel, fontsize=8)
        for line in ax.get_lines():
            line.set_clip_on(False)
            # line.set_capsize(3)

        # Disable spines.
        if not 'ax4' in axname:
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.spines['left'].set_position(('outward', 3))
            ax.spines['bottom'].set_position(('outward', 3))
        else:
            ax.yaxis.set_ticks_position('right')
            ax.spines['right'].set_position(('outward', 3))
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('black')
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_position(('outward', 3))

            ax.spines['left'].set_position(('outward', 3)) 


        ax.grid(axis='y', linewidth=0.3)

        pad = 2

        if axname == 'ax1b':
            label_1 = "Lagged"
            label_2 = "Contemp."
            label_3 = "Auto"

            ax.errorbar([], [], linestyle='',
                capsize=capsize, label=label_1,
                color='black', marker=marker1)
            ax.errorbar([], [], linestyle='',
                capsize=capsize, label=label_2,
                color='black', marker=marker2)
            ax.errorbar([], [], linestyle='',
                capsize=capsize, label=label_3,
                color='black', marker=marker3)
            ax.legend(ncol=2,
                    columnspacing=0.,
                    # bbox_to_anchor=(0., 1.02, 1., .03), borderaxespad=0, mode="expand", 
                    loc='upper right', fontsize=5, framealpha=0.3
                    ) #.draw_frame(False)
        
        if axname == 'ax1a':
            ax.set_title('Adj. TPR', fontsize=6, pad=pad)
            ax.set_ylim(0., 1.)
            # ax.spines['left'].set_position(('outward', 3))
            # ax.grid(axis='y')
            ax.tick_params(labelbottom=False)    


        elif axname == 'ax1b':
            if fpr_precision == 'precision':
                ax.set_title('Adj. precision', fontsize=6, pad=pad)
                ax.set_ylim(0., 1.)
            else:
                # ax.tick_params(labelleft=False)  
                ax.set_title('Adj. FPR', fontsize=6, pad=pad)
                if which != 'pc_alpha':
                    ax.set_yscale('symlog', linthreshy=pc_alpha*2)
                    ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.set_ylim(0., .1)

        elif axname == 'ax2a':
            ax.set_title('Orient. recall', fontsize=6, pad=pad)
            ax.set_ylim(0., 1.)
            ax.tick_params(labelbottom=False)    
            # ax.tick_params(labelleft=False)    


        elif axname == 'ax2b':
            ax.set_title('Orient. precision', fontsize=6, pad=pad)
            ax.set_ylim(0., 1.)
            # ax.tick_params(labelleft=False)    

        elif axname == 'ax3a':
            ax.set_title('Effect size', fontsize=6, pad=pad)
            ax.set_ylim(0., 0.5)
            ax.tick_params(labelbottom=False)    

        elif axname == 'ax3b':
            ax.set_title('Cardinality', fontsize=6, pad=pad)

        elif axname == 'ax4a':
            ax.set_title('Runtime [s]', fontsize=6, pad=pad)
            # ax.set_ylim(0., 1.)
            ax.tick_params(labelbottom=False)    
        
        elif axname == 'ax4b':
            ax.set_title('True PAG', fontsize=6, pad=pad)
            ax.set_ylim(0., 1.)

            label_1 = "Lagged"
            label_2 = "Contemp."
            label_3 = "Auto"

            ax.plot([], [], linestyle='',
                label=r'directed', #$\rightarrow$',
                color='black', marker='>')
            ax.plot([], [], linestyle='',
                 label=r'bidirected', #$\leftrightarrow$',
                color='black', marker='D')
            ax.plot([], [], linestyle='',
                label=r'unoriented', #$\rightarrow$',
                color='black', marker='o', fillstyle='none')
            ax.legend(ncol=1, 
                columnspacing=.5,
                    # bbox_to_anchor=(0., 1.02, 1., .03), borderaxespad=0, mode="expand", 
                    loc='upper left', fontsize=5, framealpha=0.3
                    ) 

    axlegend = fig.add_axes([0.05, .89, 1., .05])
    axlegend.axis('off')
    for method in methods:
        axlegend.errorbar([], [], linestyle='',
            capsize=capsize, label=method_label(method),
            color=color_picker(method), marker='s')
    
    # if not 'paper' in variant:
    #     ncol = 1
    #     fontsize = 5
    # else:
    ncol = 3 #len(methods)
    fontsize = 6
    axlegend.legend(ncol=ncol,
            # bbox_to_anchor=(0., 1.0, 1., .03),
             loc='lower left',
            # borderaxespad=0, mode="expand", 
            markerscale=3,
            columnspacing=.75,
            labelspacing=.01,
            fontsize=fontsize, framealpha=.5
            ) #.draw_frame(False)

    # if 'paper' in variant and SM is False:
    #     if 'autocorrfinalphases' in variant:  # and ci_test == 'par_corr':
    #         plt.figtext(0., 1., "A", fontsize=12, fontweight='bold',
    #             ha='left', va='top')
    #     elif 'autocorr' in variant:
    #         plt.figtext(0., 1., "B", fontsize=12, fontweight='bold',
    #             ha='left', va='top')
    #     elif 'highdim' in variant:
    #         plt.figtext(0., 1., "C", fontsize=12, fontweight='bold',
    #             ha='left', va='top')
    #     elif 'tau_max' in variant:
    #         plt.figtext(0., 1., "D", fontsize=12, fontweight='bold',
    #             ha='left', va='top')

    if which == 'N':
        plt.figtext(0.5, 0., r"Number of variables $N$", fontsize=8,
            horizontalalignment='center', va='bottom')
        plt.figtext(1., 1., r"$T=%d, a=%s, \tau_{\max}=%d, \lambda=%s$" %(T, auto, tau_max, frac_unobserved) 
                            +"\n" + r"%s, $\alpha=%s$" %(name[ci_test],pc_alpha),
         fontsize=6, ha='right', va='top')
    elif which == 'auto':
        plt.figtext(0.5, 0., r"Autocorrelation $a$", fontsize=8,
            horizontalalignment='center', va='bottom')
        plt.figtext(1., 1., r"$N=%d, T=%d, \tau_{\max}=%d, \lambda=%s$" %(N, T, tau_max, frac_unobserved) 
                            +"\n" + r"%s, $\alpha=%s$" %(name[ci_test], pc_alpha),
         fontsize=6, ha='right', va='top')

    elif which == 'tau_max':
        plt.figtext(0.5, 0., r"Time lag $\tau_{\max}$", fontsize=8,
            horizontalalignment='center', va='bottom')
        plt.figtext(1., 1., r"$N=%d, T=%d, a=%s, \lambda=%s$" %(N, T, auto, frac_unobserved) 
                            +"\n" + r"%s, $\alpha=%s$" %(name[ci_test], pc_alpha),
         fontsize=6, ha='right', va='top')

    elif which == 'unobserved':
        plt.figtext(0.5, 0., r"Frac. unobserved", fontsize=8,
            horizontalalignment='center', va='bottom')
        # plt.figtext(1., 1., r"$N=%d, a=%s, T=%d, \alpha=%s$" %(N, auto, T, pc_alpha),)
         # fontsize=6, ha='right', va='top')
        plt.figtext(1., 1., r"$N=%d, T=%d, a=%s, \tau_{\max}=%d$" %(N, T, auto, tau_max) 
                            +"\n" + r"%s, $\alpha=%s$" %(name[ci_test], pc_alpha),
         fontsize=6, ha='right', va='top')

    elif which == 'sample_size':
        plt.figtext(0.5, 0., r"Sample size $T$", fontsize=8,
            horizontalalignment='center', va='bottom')
        plt.figtext(1., 1., r"$N=%d, a=%s, \tau_{\max}=%d, \lambda=%s$" %(N, auto, tau_max, frac_unobserved) 
                            +"\n" + r"%s, $\alpha=%s$" %(name[ci_test], pc_alpha),
         fontsize=6, ha='right', va='top')


    fig.subplots_adjust(left=0.06, right=0.93, hspace=.3, bottom=0.12, top=0.85, wspace=.3)
    fig.savefig(save_folder + '%s.%s' %(save_suffix, save_type))
    plot_files.append(save_folder + '%s.%s' %(save_suffix, save_type))


def color_picker(method):

    # if not 'paper' in variant:
    #     colors = ['orange', 'red', 'green', 'blue', 'grey', 'lightgreen']
    #     return colors[methods.index(method)]

    if method == 'svarfci':
        return 'magenta'
    elif method == 'svarrfci':
        return 'orange'   
    elif 'lpcmci' in method:
        cmap = plt.get_cmap('Greens')
        if 'prelim0' in method:
            return cmap(0.3)
        elif 'prelim1' in method:
            return cmap(0.4)
        elif 'prelim2' in method:
            return cmap(0.5)
        elif 'prelim3' in method:
            return cmap(0.6)
        elif 'prelim4' in method:
            return cmap(0.7)
    else:
        return 'grey'


def links_from_N(num_nodes):

    if 'highdim' in variant:
        return num_nodes
    else:
        return num_nodes


if __name__ == '__main__':


    save_type = 'pdf'
    plot_files = []
    paper = False
    SM = True
    fpr_precision = 'fpr'

    # Directory to save figures
    folder_name = "results/"
    save_folder = "figures/"

    methods = [
            "lpcmci_nprelim0",
            "lpcmci_nprelim4",
            "svarfci",
            "svarrfci",
            ]


    if variant == 'autocorr':
        if ci_test == 'par_corr':
            model  = 'random_lineargaussian'  # random_lineargaussian random_nonlinearmixed
            T_here = [200, 500, 1000]
            N_here = [5] #, 10]  #[3, 5, 10]
            num_rows = 3
        else:
            model = 'random_nonlinearmixed'
            T_here = [200, 400] # [200, 500, 1000]
            N_here = [3, 5, 10] 
            num_rows = 3

        tau_max = 5
        vary_auto = [0., 0.5, 0.9, 0.95, 0.99]   # [0., 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

        pc_alpha_here = [0.01, 0.05]
        
        min_coeff = 0.2
        coeff = 0.8

        frac_unobserved = 0.3
        contemp_fraction = 0.3
        max_true_lag = 3

        for T in T_here: 
         for N in N_here:

          if N == 2: n_links = 1
          else: n_links = N  
          
          for pc_alpha in pc_alpha_here: 

            para_setup_name = (variant, N, n_links, min_coeff, coeff, contemp_fraction, frac_unobserved,  
                max_true_lag, T, ci_test, pc_alpha, tau_max) 

            save_suffix = '%s-'*len(para_setup_name) % para_setup_name
            save_suffix = save_suffix[:-1]

            print(save_suffix)
            draw_it(paras=vary_auto, which='auto')  


    elif variant == 'highdim':
        if ci_test == 'par_corr':
            model  = 'random_lineargaussian'  # random_lineargaussian random_nonlinearmixed
            T_here = [200, 500, 1000]
            vary_N =  [3, 5, 7, 10, 15]   
            auto_here = [0., 0.5, 0.95, 0.99] 
            num_rows = 4
        else:
            model = 'random_nonlinearmixed'
            T_here = [200] #, 500, 1000]
            vary_N =  [3, 5]    
            num_rows = 2

        contemp_fraction = .3
        frac_unobserved = 0.3
        max_true_lag = 3
        tau_max = 5

        min_coeff = 0.2
        coeff = 0.8
        
        for T in T_here:        
            for auto in auto_here: #, 0.5, 0.9]:
                
              for pc_alpha in [0.01, 0.05]: #, 0.1]:

                para_setup_name = (variant, min_coeff, coeff, auto, contemp_fraction, frac_unobserved, max_true_lag, T, ci_test, pc_alpha, tau_max)
                save_suffix = '%s-'*len(para_setup_name) % para_setup_name
                save_suffix = save_suffix[:-1]

                print(save_suffix)
                draw_it(paras=vary_N, which='N')  

    elif  variant == 'sample_size':

        if ci_test == 'par_corr':
            model  = 'random_lineargaussian'  # random_lineargaussian random_nonlinearmixed
            vary_T = [200, 500, 1000]
            N_here = [3, 5, 10]
            auto_here = [0., 0.5, 0.95, 0.99] 
            num_rows = 4
        else:
            model = 'random_nonlinearmixed'
            vary_T = [200, 500, 1000]
            N_here = [5]
            auto_here = [0., 0.5, 0.95, 0.99] 
            num_rows = 2

        min_coeff = 0.2
        coeff = 0.8
        
        contemp_fraction = 0.3
        frac_unobserved = 0.3

        max_true_lag = 3
        tau_max = 5

        for N in N_here:
          if N == 2: n_links = 1
          else: n_links = N   
          for auto in auto_here:
                for pc_alpha in [0.01, 0.05]: #, 0.1]:

                    para_setup_name = (variant, N, n_links, min_coeff, coeff, contemp_fraction, frac_unobserved, max_true_lag, auto, ci_test, pc_alpha, tau_max)
                    save_suffix = '%s-'*len(para_setup_name) % para_setup_name
                    save_suffix = save_suffix[:-1]

                    print(save_suffix)
                    draw_it(paras=vary_T, which='sample_size')  


    elif  variant == 'unobserved':

        if ci_test == 'par_corr':
            model  = 'random_lineargaussian'  # random_lineargaussian random_nonlinearmixed
            T_here = [200, 500, 1000]
            N_here = [5, 10]
            auto_here = [0., 0.5, 0.95, 0.99] 
            num_rows = 4
        else:
            model = 'random_nonlinearmixed'
            T_here = [200, 500, 1000]
            N_here = [5]
            auto_here = [0., 0.5, 0.95, 0.99] 
            num_rows = 2

        min_coeff = 0.2
        coeff = 0.8
        
        contemp_fraction = 0.3
        vary_frac_unobserved = [0., 0.3, 0.5]

        max_true_lag = 3
        tau_max = 5

        for N in N_here:
          if N == 2: n_links = 1
          else: n_links = N  
          for T in T_here:

            for auto in auto_here:
                for pc_alpha in [0.01, 0.05]: #, 0.1]:

                    para_setup_name = (variant, N, n_links, min_coeff, coeff, contemp_fraction, max_true_lag, auto, T, ci_test, pc_alpha, tau_max)
                    save_suffix = '%s-'*len(para_setup_name) % para_setup_name
                    save_suffix = save_suffix[:-1]

                    print(save_suffix)
                    draw_it(paras=vary_frac_unobserved, which='unobserved')  

    if variant == 'tau_max':

        if ci_test == 'par_corr':
            model  = 'random_lineargaussian'  # random_lineargaussian random_nonlinearmixed
            T_here = [200, 500, 1000]
            N_here = [5]
            auto_here = [0., 0.5, 0.95, 0.99] 
            num_rows = 4
        else:
            model = 'random_nonlinearmixed'
            T_here = [200, 500, 1000]
            N_here = [5]
            auto_here = [0., 0.5, 0.95, 0.99] 
            num_rows = 2

        min_coeff = 0.2
        coeff = 0.8
   
        contemp_fraction = 0.3
        frac_unobserved = 0.3

        max_true_lag = 3
        vary_tau_max = [3, 5, 7, 10]

        for T in T_here:
         for N in N_here:
          if N == 2: n_links = 1
          else: n_links = N   
          for auto in auto_here:
                for pc_alpha in [0.01, 0.05]: #, 0.1]:

                    para_setup_name = (variant, N, n_links, min_coeff, coeff, contemp_fraction, frac_unobserved, max_true_lag, auto, T, ci_test, pc_alpha)
                    save_suffix = '%s-'*len(para_setup_name) % para_setup_name
                    save_suffix = save_suffix[:-1]

                    print(save_suffix)
                    draw_it(paras=vary_tau_max, which='tau_max')