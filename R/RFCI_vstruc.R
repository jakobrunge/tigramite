#' RFCI's v-structure discovery procedure adapted for F2CI
#' @param suffStat List containing suffStat$data
#' @param indepTest The independence test
#' @param alpha Type I error rate for indepTest
#' @param sepset Matrix of separating sets
#' @param g.amat Graph returned from skeleton discovery procedure
#' @return List containing separating sets (sepset), n

RFCI_vstruc <- function (suffStat, indepTest, alpha, sepset, g.amat, unshTripl,
          unshVect, mix_modeling, verbose=FALSE)
{

  A <- g.amat;

  time_start = proc.time(); ## begin v-structure timing

  ## begin counts
  count = lapply( 1:ncol(A), function(.) vector("list", 2) )
  count = lapply( 1:ncol(A), function(.) count )

  start=-1;
  for (i in 1:ncol(A)){
    for (j in 1:ncol(A)) {
      count[[i]][[j]][[1]] = start;
      count[[i]][[j]][[2]] = start-1;
      start=start-2;
    }
  }
  ## end counts

  stopifnot(is.matrix(unshTripl), nrow(unshTripl) == 3)
  nT <- ncol(unshTripl)
  unfTripl <- NULL
  if (nT) {

    ## begin check unshielded triples
    checktmp <- dep.triple(suffStat, indepTest, alpha, sepset = sepset,
                           apag = g.amat, unshTripl = unshTripl, unshVect = unshVect,
                           trueVstruct = rep(TRUE, nT), verbose = verbose)
    A <- checktmp$apag
    sepset <- checktmp$sepset
    unshTripl <- checktmp$triple
    unshVect <- checktmp$vect
    trueVstruct <- checktmp$trueVstruct
    ##end check unshielded triples

    if (any(trueVstruct)) {
      for (i in 1:dim(unshTripl)[2]) {
        if (trueVstruct[i]) {
          x <- unshTripl[1, i]
          y <- unshTripl[2, i]
          z <- unshTripl[3, i]

          if ( !(y %in% sepset[[x]][[z]]) && !(y %in% sepset[[z]][[x]]) ){

            mix_res1_j = mix_modeling(x, y, sepset[[x]][[z]], suffStat, joint=TRUE);
            #mix_res1_j = mix_modeling_joint_ICL(x, y, sepset[[x]][[z]], suffStat);
            #mix_res1_j = 1;
            if (mix_res1_j == 1){
              mix_res1_s1 = 1;
              mix_res1_s2 = 1;
            } else{
              mix_res1_s = mix_modeling(x, y, sepset[[x]][[z]], suffStat, joint=FALSE);
              mix_res1_s1 = mix_res1_s[[1]];
              mix_res1_s2 = mix_res1_s[[2]];
            }

            mix_res2_j = mix_modeling(y,z,sepset[[x]][[z]],suffStat, joint=TRUE);
            #mix_res2_j = mix_modeling_joint_ICL(y,z, sepset[[x]][[z]], suffStat);
            #mix_res2_j = 1;
            if (mix_res2_j == 1){
              mix_res2_s1 = 1;
              mix_res2_s2 = 1;
            } else{
              mix_res2_s = mix_modeling(y,z,sepset[[x]][[z]],suffStat, joint=FALSE);
              mix_res2_s1 = mix_res2_s[[1]];
              mix_res2_s2 = mix_res2_s[[2]];
            }

            #print(c(x,y,z,sepset[[x]][[z]]))
            #print(c(mix_res1_j, mix_res1_s1, mix_res1_s2))
            #print(c(mix_res2_j, mix_res2_s1, mix_res2_s2))

            if (A[x, z] == 0 && A[x, y] != 0 && A[z, y] != 0 &&
                mix_res1_j==1 && mix_res2_j==1) {

              A[c(x, z), y] <- 2 ## x -> y <- z


            } else if (A[x, z] == 0 && A[x, y] != 0 && A[z, y] != 0 &&
              ((mix_res1_j==1 && xor(mix_res2_s1==1, mix_res2_s2 == 1)) || (mix_res2_j==1 && xor(mix_res1_s1==1, mix_res1_s2 == 1))) ) {

              max_count = max(unlist(count)); if (max_count < 0) {max_count = 0};

              if ( A[x,y] != 2 ) {A[x,y] = 4; count[[x]][[y]][[1]] = c(count[[x]][[y]][[1]], max_count+1);} ##x ->> y *- z
              if ( A[z,y] != 2 ) {A[z,y] = 4; count[[z]][[y]][[1]] = c(count[[z]][[y]][[1]], max_count+1);} ##x -* y <<- z

            }

          }
        }
      }
    }
  }

  time_vs = proc.time() - time_start; ## end v-structure timing

  list(sepset = sepset, G = A*1, unfTripl = unfTripl, count=count, time_vs=time_vs)
}
