#' RFCI's v-structure discovery procedure adapted for F2CI
#' @param suffStat List containing suffStat$data
#' @param indepTest The independence test
#' @param alpha Type I error rate for indepTest
#' @return List containing

IP_discovery <- function(suffStat,indepTest,alpha, max.cs){


  time_start <- proc.time();
  skel <- skeleton_new(suffStat, indepTest, alpha, p = ncol(suffStat$data), m.max=max.cs)

  G_sk <- as(skel@graph, "matrix")
  sepset_sk <- skel@sepset

  time_skel = proc.time()-time_start;

  pdsepRes <- pdsep(skel@graph, suffStat, indepTest,
                    ncol(suffStat$data), sepset_sk, alpha, skel@pMax, m.max=max.cs)

  G <- pdsepRes$G
  sepset <- pdsepRes$sepset

  time_pdsep = proc.time()-time_start;

  resIP <- list(G=G,G_sk=G_sk, sepset=sepset, sepset_sk = sepset_sk, time_skel=time_skel, time_pdsep=time_pdsep, skel=skel, pdsepRes=pdsepRes)

  return(resIP)

}
