generate_samps_DAG <- function(nsamps,means,weights){

  weights = t(weights);

  p = nrow(weights);

  N = (p*p - p)/2;

  matrix_s = solve(diag(p)-weights);

  samps = mvrnorm(nsamps,means,matrix_s %*% t(matrix_s));

  return(samps);


  }
