import numpy as np

def get_masks(true_graphs):

    n_realizations, N, N, taumaxplusone = true_graphs.shape
    tau_max = taumaxplusone - 1

    cross_mask = np.repeat(np.identity(N).reshape(N,N,1)==False, tau_max + 1, axis=2).astype('bool')
    cross_mask[range(N),range(N),0]=False
    contemp_cross_mask_tril = np.zeros((N,N,tau_max + 1)).astype('bool')
    contemp_cross_mask_tril[:,:,0] = np.tril(np.ones((N, N)), k=-1).astype('bool')

    lagged_mask = np.ones((N,N,tau_max + 1)).astype('bool')
    lagged_mask[:,:,0] = 0
    # auto_mask = np.ones((N,N,tau_max + 1)).astype('bool')
    auto_mask = lagged_mask*(cross_mask == False)

    any_mask = np.ones((N,N,tau_max + 1)).astype('bool')
    any_mask[:,:,0] = contemp_cross_mask_tril[:,:,0]

    # n_realizations = len(results['graphs'])
    # true_graphs = results['true_graphs']

    cross_mask = np.repeat(cross_mask.reshape(1, N,N,tau_max + 1), n_realizations, axis=0)
    contemp_cross_mask_tril = np.repeat(contemp_cross_mask_tril.reshape(1, N,N,tau_max + 1), n_realizations, axis=0)
    lagged_mask = np.repeat(lagged_mask.reshape(1, N,N,tau_max + 1), n_realizations, axis=0)
    auto_mask = np.repeat(auto_mask.reshape(1, N,N,tau_max + 1), n_realizations, axis=0)
    any_mask = np.repeat(any_mask.reshape(1, N,N,tau_max + 1), n_realizations, axis=0)

    return cross_mask, contemp_cross_mask_tril, lagged_mask, auto_mask, any_mask, tau_max

def _get_match_score(true_link, pred_link):
    if true_link == "" or pred_link == "": return 0
    count = 0
    # If left edgemark is correct add 1
    if true_link[0] == pred_link[0]:
        count += 1
    # If right edgemark is correct add 1
    if true_link[2] == pred_link[2]:
        count += 1
    return count
match_func = np.vectorize(_get_match_score, otypes=[int]) 


def _get_conflicts(pred_link):
    if pred_link == "": return 0
    count = 0
    # If left edgemark is conflict add 1
    if pred_link[0] == 'x':
        count += 1
    # If right edgemark is conflict add 1
    if pred_link[2] == 'x':
        count += 1
    return count
conflict_func = np.vectorize(_get_conflicts, otypes=[int]) 

def _get_unoriented(true_link):
    if true_link == "": return 0
    count = 0
    # If left edgemark is unoriented add 1
    if true_link[0] == 'o':
        count += 1
    # If right edgemark is unoriented add 1
    if true_link[2] == 'o':
        count += 1
    return count
unoriented_func = np.vectorize(_get_unoriented, otypes=[int]) 

def get_numbers(metrics, orig_true_graphs, orig_pred_graphs, val_min, cardinality, computation_time, boot_samples=200):


    cross_mask, contemp_cross_mask_tril, lagged_mask, auto_mask, any_mask, tau_max = get_masks(orig_true_graphs)

    n_realizations = len(orig_pred_graphs)

    metrics_dict = {}

    pred_graphs = orig_pred_graphs
    true_graphs = orig_true_graphs

    metrics_dict['valmin_lagged'] = (((true_graphs!="")*np.abs(val_min)*cross_mask*lagged_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)) )
    metrics_dict['valmin_auto'] = (((true_graphs!="")*np.abs(val_min)*auto_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*auto_mask).sum(axis=(1,2,3)) )
    metrics_dict['valmin_contemp'] = (((true_graphs!="")*np.abs(val_min)*contemp_cross_mask_tril).sum(axis=(1,2,3)),
                        ((true_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)) )
    metrics_dict['valmin_anylink'] = (((true_graphs!="")*np.abs(val_min)*any_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*any_mask).sum(axis=(1,2,3)) ) 

    metrics_dict['cardinality_lagged'] = (((true_graphs!="")*cardinality*cross_mask*lagged_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)) )
    metrics_dict['cardinality_auto'] = (((true_graphs!="")*cardinality*auto_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*auto_mask).sum(axis=(1,2,3)) )
    metrics_dict['cardinality_contemp'] = (((true_graphs!="")*cardinality*contemp_cross_mask_tril).sum(axis=(1,2,3)),
                        ((true_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)) )
    metrics_dict['cardinality_anylink'] = (((true_graphs!="")*cardinality*any_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*any_mask).sum(axis=(1,2,3)) ) 

    metrics_dict['num_links_lagged'] = (((true_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)),
                        (cross_mask*lagged_mask).sum(axis=(1,2,3)) )
    metrics_dict['num_links_auto'] = (((true_graphs!="")*auto_mask).sum(axis=(1,2,3)),
                        (auto_mask).sum(axis=(1,2,3)) )
    metrics_dict['num_links_contemp'] = (((true_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)),
                        (contemp_cross_mask_tril).sum(axis=(1,2,3)) )
    metrics_dict['num_links_anylink'] = (((true_graphs!="")*any_mask).sum(axis=(1,2,3)),
                        (any_mask).sum(axis=(1,2,3)) ) 

    metrics_dict['directed_lagged'] = (((true_graphs=="-->")*cross_mask*lagged_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)) )
    metrics_dict['directed_auto'] = (((true_graphs=="-->")*auto_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*auto_mask).sum(axis=(1,2,3)) )
    metrics_dict['directed_contemp'] = ((np.logical_or(true_graphs=="-->", true_graphs=="<--")*contemp_cross_mask_tril).sum(axis=(1,2,3)),
                        ((true_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)) )
    metrics_dict['directed_anylink'] = ((np.logical_or(true_graphs=="-->", true_graphs=="<--")*any_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*any_mask).sum(axis=(1,2,3)) ) 

    metrics_dict['bidirected_lagged'] = (((true_graphs=="<->")*cross_mask*lagged_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)) )
    metrics_dict['bidirected_auto'] = (((true_graphs=="<->")*auto_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*auto_mask).sum(axis=(1,2,3)) )
    metrics_dict['bidirected_contemp'] = (((true_graphs=="<->")*contemp_cross_mask_tril).sum(axis=(1,2,3)),
                        ((true_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)) )
    metrics_dict['bidirected_anylink'] = (((true_graphs=="<->")*any_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*any_mask).sum(axis=(1,2,3)) ) 

    # Adjacency true/false positives and precision/recall, separated by lagged/auto/contemp
    metrics_dict['adj_lagged_fpr'] = ( ((true_graphs=="")*(pred_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)),  
                                          ((true_graphs=="")*cross_mask*lagged_mask).sum(axis=(1,2,3)) )
    metrics_dict['adj_lagged_tpr'] = (((true_graphs!="")*(pred_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)) )
    metrics_dict['adj_auto_fpr'] = (((true_graphs=="")*(pred_graphs!="")*auto_mask).sum(axis=(1,2,3)),
                        ((true_graphs=="")*auto_mask).sum(axis=(1,2,3)) )
    metrics_dict['adj_auto_tpr'] = (((true_graphs!="")*(pred_graphs!="")*auto_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*auto_mask).sum(axis=(1,2,3)) )
    metrics_dict['adj_contemp_fpr'] = (((true_graphs=="")*(pred_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)),
                        ((true_graphs=="")*contemp_cross_mask_tril).sum(axis=(1,2,3)) )
    metrics_dict['adj_contemp_tpr'] = (((true_graphs!="")*(pred_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)),
                        ((true_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)) )

    metrics_dict['adj_anylink_fpr'] = (((true_graphs=="")*(pred_graphs!="")*any_mask).sum(axis=(1,2,3)),
                        ((true_graphs=="")*any_mask).sum(axis=(1,2,3)) )
    metrics_dict['adj_anylink_tpr'] = (((true_graphs!="")*(pred_graphs!="")*any_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*any_mask).sum(axis=(1,2,3)) )            


    metrics_dict['adj_lagged_precision'] = (((true_graphs!="")*(pred_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)),
                        ((pred_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)) )
    metrics_dict['adj_lagged_recall'] = (((true_graphs!="")*(pred_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)) )
    metrics_dict['adj_auto_precision'] = (((true_graphs!="")*(pred_graphs!="")*auto_mask).sum(axis=(1,2,3)),
                        ((pred_graphs!="")*auto_mask).sum(axis=(1,2,3)) )
    metrics_dict['adj_auto_recall'] = (((true_graphs!="")*(pred_graphs!="")*auto_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*auto_mask).sum(axis=(1,2,3)) )
    metrics_dict['adj_contemp_precision'] = (((true_graphs!="")*(pred_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)),
                        ((pred_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)) )
    metrics_dict['adj_contemp_recall'] = (((true_graphs!="")*(pred_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)),
                        ((true_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)) )

    metrics_dict['adj_anylink_precision'] = (((true_graphs!="")*(pred_graphs!="")*any_mask).sum(axis=(1,2,3)),
                        ((pred_graphs!="")*any_mask).sum(axis=(1,2,3)) )
    metrics_dict['adj_anylink_recall'] = (((true_graphs!="")*(pred_graphs!="")*any_mask).sum(axis=(1,2,3)),
                        ((true_graphs!="")*any_mask).sum(axis=(1,2,3)) )


    # Edge mark precision and recall
    metrics_dict['edgemarks_lagged_precision'] = ((match_func(true_graphs,
                                                               pred_graphs)*(cross_mask*lagged_mask)).sum(axis=(1,2,3)),
                                                        2.*((pred_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)) )

    metrics_dict['edgemarks_lagged_recall'] = ((match_func(true_graphs, pred_graphs)*(cross_mask*lagged_mask)).sum(axis=(1,2,3)),
                                                        2.*((true_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)) )
    metrics_dict['edgemarks_auto_precision'] = ((match_func(true_graphs, pred_graphs)*auto_mask).sum(axis=(1,2,3)),
                                                        2.*((pred_graphs!="")*auto_mask).sum(axis=(1,2,3)) )
    metrics_dict['edgemarks_auto_recall'] = ((match_func(true_graphs, pred_graphs)*auto_mask).sum(axis=(1,2,3)),
                                                        2.*((true_graphs!="")*auto_mask).sum(axis=(1,2,3)) )
    metrics_dict['edgemarks_contemp_precision'] = ((match_func(true_graphs, pred_graphs)*contemp_cross_mask_tril).sum(axis=(1,2,3)),
                                                        2.*((pred_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)) )
    metrics_dict['edgemarks_contemp_recall'] = ((match_func(true_graphs, pred_graphs)*contemp_cross_mask_tril).sum(axis=(1,2,3)),
                                                        2.*((true_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)) )

    metrics_dict['edgemarks_anylink_precision'] = ((match_func(true_graphs, pred_graphs)*any_mask).sum(axis=(1,2,3)),
                                                        2.*((pred_graphs!="")*any_mask).sum(axis=(1,2,3)) )
    metrics_dict['edgemarks_anylink_recall'] = ((match_func(true_graphs, pred_graphs)*any_mask).sum(axis=(1,2,3)),
                                                        2.*((true_graphs!="")*any_mask).sum(axis=(1,2,3)) )

    # Unoriented marks in true_graph and conflicts in pred_graph
    metrics_dict['unoriented_lagged'] = ((unoriented_func(true_graphs)*(cross_mask*lagged_mask)).sum(axis=(1,2,3)),
                                                        2.*((true_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)) )
    metrics_dict['conflicts_lagged'] = ((conflict_func(pred_graphs)*(cross_mask*lagged_mask)).sum(axis=(1,2,3)),
                                                        2.*((pred_graphs!="")*cross_mask*lagged_mask).sum(axis=(1,2,3)) )
    metrics_dict['unoriented_auto'] = ((unoriented_func(true_graphs)*(auto_mask)).sum(axis=(1,2,3)),
                                                        2.*((true_graphs!="")*auto_mask).sum(axis=(1,2,3)) )
    metrics_dict['conflicts_auto'] = ((conflict_func(pred_graphs)*(auto_mask)).sum(axis=(1,2,3)),
                                                        2.*((pred_graphs!="")*auto_mask).sum(axis=(1,2,3)) )
    metrics_dict['unoriented_contemp'] = ((unoriented_func(true_graphs)*(contemp_cross_mask_tril)).sum(axis=(1,2,3)),
                                                        2.*((true_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)) )
    metrics_dict['conflicts_contemp'] = ((conflict_func(pred_graphs)*(contemp_cross_mask_tril)).sum(axis=(1,2,3)),
                                                        2.*((pred_graphs!="")*contemp_cross_mask_tril).sum(axis=(1,2,3)) )

    metrics_dict['unoriented_anylink'] = ((unoriented_func(true_graphs)*(any_mask)).sum(axis=(1,2,3)),
                                                        2.*((true_graphs!="")*any_mask).sum(axis=(1,2,3)) )
    metrics_dict['conflicts_anylink'] = ((conflict_func(pred_graphs)*(any_mask)).sum(axis=(1,2,3)),
                                                            2.*((pred_graphs!="")*any_mask).sum(axis=(1,2,3)) )
    
    for metric in metrics_dict.keys():

        numerator, denominator = metrics_dict[metric]

        metric_boot = np.zeros(boot_samples)
        for b in range(boot_samples):
            # Store the unsampled values in b=0
            rand = np.random.randint(0, n_realizations, n_realizations)
            metric_boot[b] = numerator[rand].sum()/denominator[rand].sum()

        metrics_dict[metric] = (numerator.sum()/denominator.sum(), metric_boot.std())

    metrics_dict['computation_time'] = (np.mean(np.array(computation_time)), np.percentile(np.array(computation_time), [5, 95]))

    return metrics_dict

def get_evaluation(results, from_file=False):

    metrics = [ 'adj_' + link_type + "_" + metric_type for link_type in ['lagged', 'auto', 'contemp', 'anylink'] 
                                                       for metric_type in ['fpr', 'tpr']]
    metrics += [ 'adj_' + link_type + "_" + metric_type for link_type in ['lagged', 'auto', 'contemp', 'anylink'] 
                                                       for metric_type in ['precision', 'recall']]
    metrics +=  [ 'edgemarks_' + link_type + "_" + metric_type for link_type in ['lagged', 'auto', 'contemp', 'anylink'] 
                                                       for metric_type in ['precision', 'recall']]
    metrics +=  [ metric_type + "_" + link_type for link_type in ['lagged', 'auto', 'contemp', 'anylink'] 
                                for metric_type in ['unoriented', 'conflicts', 'num_links', 'directed', 'bidirected']]

    metrics += [ 'valmin_' + link_type for link_type in ['lagged', 'auto', 'contemp', 'anylink']]
    metrics += [ 'cardinality_' + link_type for link_type in ['lagged', 'auto', 'contemp', 'anylink']]

    metrics += ['computation_time']

    if results is not None:

        # all_configs[conf]['graphs'][i] = all_configs[conf]['results'][i]['graph']
        # all_configs[conf]['true_graphs'][i] = all_configs[conf]['results'][i]['true_graph']
        # all_configs[conf]['computation_time'].append(all_configs[conf]['results'][i]['computation_time'])

        # Same tau_max for all trials
        orig_true_graphs = results['true_graphs']

        # Minimum effect size for each link
        val_min = results['val_min']

        # Maximum condition cardinality for each link
        cardinality = results['max_cardinality']

        # Pred graphs also contain 2's for conflicting links...
        orig_pred_graphs = results['graphs']

        computation_time = results['computation_time']
        # print(true_graphs.shape, pred_graphs.shape, contemp_cross_mask.shape, cross_mask.shape, lagged_mask.shape, (cross_mask*lagged_mask).shape )

        metrics_dict = get_numbers(metrics, orig_true_graphs, orig_pred_graphs, val_min, cardinality, computation_time)
        return metrics_dict
    else:
        return None