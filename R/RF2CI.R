#' The RF2CI algorithm
#' @param suffStat List of all information required for the conditional independence test
#' @param indepTest A conditional independence test function internally called as indepTest(x,y,z,suffStat). Tests conditional independence of x and y given z.
#' @param alpha Alpha value for conditional independence test
#' @param max.cs Maximum conditioning set size
#' @param SK_results Skeleton results if skeleton computed elsewhere
#' @return A list containing the results obtained after the skeleton, v_structure and orientation rule procedures. Main graph output in results$orientation_rules$G
#'
RF2CI <- function(suffStat,indepTest,alpha, mix_modeling, max.cs, SK_results=NULL){

  if (length(SK_results)==0){
    skel <- skeleton_new(suffStat, indepTest, alpha, p = ncol(suffStat$data), max.cs )

    SK_results = list();

    SK_results$G <- (as(skel@graph, "matrix")==1)
    SK_results$sepset <- skel@sepset;
  } else{

    SK_results$G <- SK_results$G_sk==1;
    SK_results$sepset <- SK_results$sepset_sk;
  }



  ## begin v-structure discovery
  sk.A <- SK_results$G
  sepset <- SK_results$sepset

  u.t <- find.unsh.triple(sk.A, check = FALSE)
  VS_results <- RFCI_vstruc(suffStat, indepTest, alpha, sepset, sk.A,
                            unshTripl = u.t$unshTripl, unshVect = u.t$unshVect, mix_modeling)
  ## end v-structure discovery



  ## begin orientation rules

  rules = rep(TRUE,10);
  rules[5] = FALSE;

  OR_results <- OR_discovery2(VS_results$G, suffStat, CI_test, alpha, mix_modeling, VS_results$sepset,
                              VS_results$count, rules=rules, verbose=FALSE);
  ## end orientation rules


  time_tot = SK_results$time_pdsep + VS_results$time_vs + OR_results$time_or;

  results = list(skeleton = SK_results, v_structures = VS_results, orientation_rules = OR_results,
                 time = time_tot);

  return(results)

}
