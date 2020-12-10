import numpy as np
from numpy.random import binomial
from scipy.special import expit
from tigramite.data_processing import Graph

def binomial_scp(links, T, n_binom, random_state = None, extremity = 4/5, scale = 1/2, 
    centralize = True, cut_off = True):

    if random_state is None:
        random_state = np.random.RandomState(None)

    N = len(links.keys())
    
    # Check parameters
    if type(n_binom) != int or n_binom < 2 or n_binom % 2 != 0:
        raise ValueError("n_binom must be a positive even integer")

    max_lag = 0
    contemp_dag = Graph(N)
    for j in range(N):
        for ((var, lag), coeff, func) in links[j]:
            if lag == 0:
                contemp = True
            if var not in range(N):
                raise ValueError("var must be in 0..{}.".format(N-1))
            if 'float' not in str(type(coeff)):
                raise ValueError("coeff must be float.")
            if lag > 0 or type(lag) != int:
                raise ValueError("lag must be non-positive int.")
            max_lag = max(max_lag, abs(lag))

            # Create contemp DAG
            if var != j and lag == 0:
                contemp_dag.addEdge(var, j)

    if contemp_dag.isCyclic() == 1: 
        raise ValueError("Contemporaneous links must not contain cycle.")
        
    causal_order = contemp_dag.topologicalSort() 

    transient = int(.2*T)

    data = np.zeros((T+transient, N), dtype='int')
    cut_off_value = n_binom/2
    
    for t in range(max_lag):
        for j in causal_order:

            p_add_logit_half = sum([coeff*func(data[t + lag, var]) for ((var, lag), coeff, func) in links[j] if t + lag >= 0])
            p_binom = 1/2 + (expit(p_add_logit_half*scale*4/N) - 1/2)*extremity

            if centralize:
                data[t, j] = np.rint(random_state.binomial(n_binom, p_binom) - n_binom*p_binom)
            else:
                data[t, j] = random_state.binomial(n_binom, p_binom)

            if cut_off and abs(data[t, j]) > cut_off_value:
                data[t, j] = np.sign(data[t, j])*cut_off_value

    for t in range(max_lag, T + transient):
        for j in causal_order:

            p_add_logit_half = sum([coeff*func(data[t + lag, var]) for ((var, lag), coeff, func) in links[j]])
            p_binom = 1/2 + (expit(p_add_logit_half*scale*4/N) - 1/2)*extremity

            if centralize:
                data[t, j] = np.rint(random_state.binomial(n_binom, p_binom) - n_binom*p_binom)
            else:
                data[t, j] = random_state.binomial(n_binom, p_binom)

            if cut_off and abs(data[t, j]) > cut_off_value:
                data[t, j] = np.sign(data[t, j])*cut_off_value
            
    data = data[transient:]

    return data, False

def discretized_scp(links, T, n_binom, random_state = None, centralize = True, cut_off = True):

    if random_state is None:
        random_state = np.random.RandomState(None)
        
    N = len(links.keys())

    # Check parameters
    if type(n_binom) != int or n_binom < 2 or n_binom % 2 != 0:
        raise ValueError("n_binom must be a positive even integer")

    # Prepare noise functions
    if centralize:
        noises = [lambda n_samples: random_state.binomial(n_binom, 0.5, n_samples) - n_binom*0.5 for k in range(N)]
    else:
        noises = [lambda n_samples: random_state.binomial(n_binom, 0.5, n_samples) for k in range(N)]

    # Check parameters
    max_lag = 0
    contemp_dag = Graph(N)
    for j in range(N):
        for ((var, lag), coeff, func) in links[j]:
            if lag == 0:
                contemp = True
            if var not in range(N):
                raise ValueError("var must be in 0..{}.".format(N-1))
            if 'float' not in str(type(coeff)):
                raise ValueError("coeff must be float.")
            if lag > 0 or type(lag) != int:
                raise ValueError("lag must be non-positive int.")
            max_lag = max(max_lag, abs(lag))

            # Create contemp DAG
            if var != j and lag == 0:
                contemp_dag.addEdge(var, j)

    if contemp_dag.isCyclic() == 1: 
        raise ValueError("Contemporaneous links must not contain cycle.")

    causal_order = contemp_dag.topologicalSort() 

    transient = int(.2*T)

    cut_off_value = n_binom/2

    data = np.zeros((T+transient, N), dtype='int')
    for j in range(N):
        data[:, j] = noises[j](T+transient)

    for t in range(max_lag, T+transient):
        for j in causal_order:
            increment = np.rint(sum([coeff*func(data[t + lag, var]) for ((var, lag), coeff, func) in links[j]]))
            data[t, j] += increment

            if cut_off and abs(data[t, j]) > cut_off_value:
                data[t, j] = np.sign(data[t, j])*cut_off_value

    data = data[transient:]

    nonstationary = (np.any(np.isnan(data)) or np.any(np.isinf(data)))

    return data, nonstationary
