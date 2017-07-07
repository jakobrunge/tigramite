#' Joint (bivariate) Gaussian conditional mixture modeling
#' @param a First non-conditioning variable number
#' @param b Second non-conditioning variable number
#' @param cond_set Conditioning set variable numbers
#' @return mix_res Single positive integer specifying number of mixtures for p(a,b|cond_set)

gauss_mix_modeling_joint <- function(a, b, cond_set, suffStat){

  # mix_res = mix_modeling_single(a, b, cond_set, suffStat);
  #
  # if (mix_res[[1]]>1 || mix_res[[2]]>1){
  #   mix_res = 2;
  # } else{
  #   mix_res = 1;
  # }

   data=suffStat$data;

   mix_res = try({

     if (length(cond_set) == 0) {
       mix_res = stepFlexmix(data[,c(a,b)]~1, k=1:3, nrep=3, verbose=FALSE, model=FLXMCmvnorm(diagonal=FALSE)); # mixture modeling x | cond_set(x,z)
       #mix_res = stepFlexmix(data[,c(a,b)]~1, k=1:3, nrep=3, verbose=FALSE, model=FLXMRglm3());
     } else{
       mix_res = stepFlexmix(data[,c(a,b)]~data[,cond_set], k=1:3, nrep=3, verbose=FALSE, model=FLXMRglm3()); # mixture modeling x | cond_set(x,z)
     }

       # mix_res@models$`1`@df = mix_res@models$`1`@df;
       # mix_res@models$`2`@df = mix_res@models$`2`@df;
       # mix_res@models$`3`@df = mix_res@models$`3`@df;

     mix_res = match( min(BIC(mix_res)), BIC(mix_res) );
     if (mix_res > 1) {mix_res = 2;}
     mix_res;
   })

   if (class(mix_res)=="try-error"){  mix_res = 2; }

  return(mix_res)

}
