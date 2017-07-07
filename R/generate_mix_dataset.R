#' Generates a mixture dataset from a Gaussian CMJ. Samples time uniformly from beginning to end of CMJ
#' @param CMJ The Gaussian CMJ
#' @param nSamples Number of samples desired
#'
generate_mix_dataset <- function(CMJ, nSamples){

  prop_S = CMJ$propS;

  if (length(prop_S)==0){nSamples_u = nSamples;
  } else {
  nSamples_u = nSamples*(1/min(prop_S))*length(CMJ$S)*2;  #upper bound on number of samples
  }
  nSamples_u = ceiling(nSamples_u);

  samps_times = runif( nSamples_u,0,CMJ$times[ length(CMJ$times) ]+0.5 );

  data_t = generate_samps_DAG(nSamples_u,CMJ$graph$means,CMJ$graph$weights)

  data = matrix(0,nrow(data_t), ncol(data_t)-length(CMJ$D));

  for (t in seq_len(nSamples_u)){
    indices_n = CMJ$indices[ CMJ$times < samps_times[t] ];

    idx_n = unname(tapply(seq_along(indices_n), indices_n, max)); #find last matching element in vector

    data[t,] = data_t[ t, idx_n ];


  }

  ind = numeric();
  for (j in seq_len(length(CMJ$S)) ){

    quant_j = quantile( data[ ,CMJ$S[j] ],prop_S[j] );

    ind = union(   ind, which( (data[ ,CMJ$S[j] ]<quant_j) %in% TRUE)   );


  }

  if (length(ind) != 0){
  data = data[-ind,];
  }

  if (length(union(CMJ$L,CMJ$S))>0){
    data = data[,-c(CMJ$L,CMJ$S)];
  }

  data = data[1:nSamples,];

  return(data)

}
