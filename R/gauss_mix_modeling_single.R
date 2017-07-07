#' Univariate Gaussian conditional mixture modeling
#' @param a First non-conditioning variable number
#' @param b Second non-conditioning variable number
#' @param cond_set Conditioning set variable numbers
#' @return mix_res A list containing two elements. The first element is a single positive integer for the number of mixtures for p(a|cond_set); likewise for the second element for p(b|cond_set).


gauss_mix_modeling_single <- function(a, b, cond_set, suffStat){

  data=suffStat$data;

  #if (length(cond_set) == 0) {
  # mix_res_a = stepFlexmix(data[,a]~1, k=1:3, nrep=1, verbose=FALSE); # mixture modeling x | cond_set(x,z)
  # mix_res_c = stepFlexmix(data[,c]~1, k=1:3, nrep=1, verbose=FALSE); # mixture modeling y | cond_set(x,z)
  #} else{
  #mix_res_a = stepFlexmix(data[,a]~data[,cond_set], k=1:3, nrep=1, verbose=FALSE); # mixture modeling x | cond_set(x,z)
  # mix_res_c = stepFlexmix(data[,c]~data[,cond_set], k=1:3, nrep=1, verbose=FALSE); # mixture modeling y | cond_set(x,z)
  #}

  #mix_res_1 = match( min(BIC(mix_res_a)), BIC(mix_res_a) );
  # mix_res_2 = match( min(BIC(mix_res_c)), BIC(mix_res_c) );

  #if (mix_res_1 > 1) {mix_res_1 = 2;}
  #if (mix_res_2 > 1) {mix_res_2 = 2;}

  #mix_results = list(mix_res_1 = mix_res_1, mix_res_2 = mix_res_2);

  #return(mix_results)

  #error = {mix_results = list(mix_res_1 = 2, mix_res_2 = 2);
  mix_res_s1 = try({

    if (length(cond_set) == 0) {
      mix_res_a = stepFlexmix(data[,a]~1, k=1:3, nrep=3, verbose=FALSE); # mixture modeling x | cond_set(x,z)

    } else{
      mix_res_a = stepFlexmix(data[,a]~data[,cond_set], k=1:3, nrep=3, verbose=FALSE); # mixture modeling x | cond_set(x,z)

    }

    # mix_res_a@models$`1`@df = 1*mix_res_a@models$`1`@df;
    # mix_res_a@models$`2`@df = 1*mix_res_a@models$`2`@df;
    # mix_res_a@models$`3`@df = 1*mix_res_a@models$`3`@df;


    mix_res_s1 = match( min(BIC(mix_res_a)), BIC(mix_res_a) );
    if (mix_res_s1 > 1) {mix_res_s1 = 2;}
    mix_res_s1;
  })

  if (class(mix_res_s1)=="try-error"){  mix_res_s1 = 2; }


  mix_res_s2 = try({

    if (length(cond_set) == 0) {
      mix_res_b = stepFlexmix(data[,b]~1, k=1:3, nrep=3, verbose=FALSE); # mixture modeling x | cond_set(x,z)

    } else{
      mix_res_b = stepFlexmix(data[,b]~data[,cond_set], k=1:3, nrep=3, verbose=FALSE); # mixture modeling x | cond_set(x,z)

    }

    # mix_res_b@models$`1`@df = 1*mix_res_b@models$`1`@df;
    # mix_res_b@models$`2`@df = 1*mix_res_b@models$`2`@df;
    # mix_res_b@models$`3`@df = 1*mix_res_b@models$`3`@df;

    mix_res_s2 = match( min(BIC(mix_res_b)), BIC(mix_res_b) );
    if (mix_res_s2 > 1) {mix_res_s2 = 2;}
    mix_res_s2;
  });


  if (class(mix_res_s2)=="try-error"){  mix_res_s2 = 2; }

  mix_res = list(mix_res_s1 = mix_res_s1, mix_res_s2 = mix_res_s2);

  return(mix_res)

}
