#' Instantiates a Gaussian CMJ
#' @param p Number of total variables
#' @param en Expected neighborhood size
#' @param nS Number of selection variables
#' @param nL Number of latent variables
#' @param nD Number of dynamic variables, or variables with more than one jump point

instantiate_CMJ <- function(p,en, nS,nL,nD){

  L_t =  numeric();
  S_t =  numeric();

  if (nS==0 & nL==0){
    graph = generate_DAG(p,en);

    nD_new = sample(1:(p-nD),nD, replace=TRUE); #randomly select dynamic variable indices
    var_indices = c(1:(p-nD),nD_new); #make the last vertices dynamic
    adj_t = graph$adj[1:(p-nD),1:(p-nD)]; #modify adjacency matrix for non-dynamic variables

  } else{

    while( (length(L_t) != nL) || (length(S_t) != nS) ){
      graph = generate_DAG(p,en);

      nD_new = sample(1:(p-nD),nD, replace=TRUE); #randomly select dynamic variable indices
      var_indices = c(1:(p-nD),nD_new); #make the last vertices dynamic
      adj_t = graph$adj[1:(p-nD),1:(p-nD)]; #modify adjacency matrix for non-dynamic variables


      L_indices_all =  which(rowSums(adj_t)>1 %in% TRUE);
      S_indices_all =  setdiff(which(colSums(adj_t)>1 %in% TRUE), nD_new); #non-dynamic seletion variables

      try({L_t = L_indices_all[ sample( length(L_indices_all), nL ) ];

      S_indices_all = setdiff(S_indices_all, L_t);

      S_t = S_indices_all[ sample( length(S_indices_all), nS ) ];
      }, silent=TRUE)


    }

  }

  graph_sel = matrix(0,p+nS,p+nS);
  graph_sel[seq_len(p),seq_len(p)]=graph$adj;
  for (s in seq_len(nS)){
    graph_sel[S_t[s],p+s]=1;
  }

  times = cumsum( c(rep(0,p-nD),runif(nD,0.1,0.5)) );

  data_indices = setdiff(unique(var_indices), c(L_t,S_t));

  propS = 0.4*runif(nS)+0.1;

  CMJ = list(graph =graph, L = L_t, S = S_t, D = nD_new, indices = var_indices,
                  times = times, indices_data = data_indices, graph_sel=graph_sel,
                  propS=propS)

  return(CMJ)

}
