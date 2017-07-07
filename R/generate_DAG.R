generate_DAG <- function(p,e_n){


  N = (p*p - p)/2;

  samplesB = rbinom(N,1, e_n/(p-1) );

  samplesU = runif(N,-1.3,1.5); #weights between [-1,-0.1] and [0.1, 1]
  samplesU[samplesU<0.1] = samplesU[samplesU<0.1]-0.2; #weights between [-1,-0.1] and [0.1, 1]

  samples = samplesB * samplesU;

  adj = matrix(0,p,p);
  adj[upper.tri(adj, diag=FALSE)] <- samplesB;

  weights = matrix(0,p,p);
  weights[upper.tri(weights, diag=FALSE)] <- samples;

  means = rnorm(p)*2;

  graph = list(adj = adj, weights = weights, means = means);
  return(graph)

}
