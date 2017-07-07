#' Gaussian conditional mixture modeling wrapper
#' @param a First non-conditioning variable number
#' @param b Second non-conditioning variable number
#' @param cond_set Conditioning set variable numbers
#' @param joint Set to TRUE or FALSE for bivariate or univariate conditional mixture modeling, respectively
#' @return mix_res The mixture results; a positive integer in the bivariate case, or a list with two elements in the univariate case

gauss_mix_modeling <- function(a, b, cond_set, suffStat, joint){

  if (joint == TRUE){
    mix_res = gauss_mix_modeling_joint(a, b, cond_set, suffStat);
  } else{
    mix_res = gauss_mix_modeling_single(a, b, cond_set, suffStat);
  }

  return(mix_res)

}
