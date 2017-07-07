#' Find all uncovered paths
#' @param p Number of variables
#' @param pag The graph in matrix form
#' @param a A vertex
#' @param b A vertex
#' @param c A vertex
#' @return List containing all paths and associated mix_mat 3D arrays

allUncovPath <- function (p, pag, a, b, c, suffStat, sepset, mix_modeling, mix_mat)
{
  visited <- rep(FALSE, p)
  visited[c(a,b,c)] <- TRUE

  indD_t <- which(pag[b,]!=0 & pag[,b]!=0 & pag[,a]==0 & pag[a,]==0 & !visited)
  indD_t = setdiff(indD_t,a);

  indD = vector();

  for (d in indD_t){
    if (b %in% sepset[[a]][[d]] ) {indD = union(indD, d)}
  }

  Ln=1;
  results=list();
  LL=list();

  if (length(indD) > 0) {
    path.list <- updateList_m(b, indD, NULL)


    while ( (length(path.list) > 0)) {

      cnt=0;
      mpath_t <- path.list[[1]]
      m <- length(mpath_t)
      d <- mpath_t[m]
      b_t <- mpath_t[m-1]

      path.list[[1]] <- NULL
      visited <- rep(FALSE, p)
      visited[c(a,mpath_t)] <- TRUE
      uncov <- TRUE

      if ( pag[d,c]!=0 & pag[c,d]!=0 & (d %in% sepset[[b_t]][[c]]) & uncov) {

        mpath <- c(a, mpath_t, c)
        n <- length(mpath)

        for (l in seq_len(n-2) ) {

          r = l+2; s = l+1;

          if (pag[mpath[l], mpath[r]] == 0 && pag[mpath[r], mpath[l]] == 0  && mpath[s] %in% sepset[[mpath[l]]][[mpath[r]]]){

            if (mix_mat[mpath[l], mpath[r], mpath[s]] == 0 | mix_mat[mpath[r], mpath[l], mpath[s]] == 0){
              mix_mat[mpath[l], mpath[r], mpath[s]] = mix_modeling_all(mpath[l], mpath[r], setdiff(sepset[[mpath[l]]][[mpath[r]]], mpath[s]), suffStat,mix_modeling)
              mix_mat[mpath[r], mpath[l], mpath[s]] = mix_mat[mpath[l], mpath[r], mpath[s]];
            }

            if (mix_mat[mpath[l], mpath[r], mpath[s]] == 2) {cnt = cnt + 1;
                if (cnt >1){uncov <- FALSE; break}
            }

          } else {uncov <- FALSE; break}

        }

        if (uncov){

          LL[[Ln]] <- list(path = mpath, cnt = cnt);
          Ln=Ln+1;
        }
      }
      else {

        indR_t <- which(pag[d,]!=0 & pag[,d]!=0 & !visited)

        indR <- which(sapply(sepset[][[b_t]], function(x) d %in% x))
        #indR = grep(d, sepset[][[b_t]])

        indR = intersect(indR,indR_t);

        if (length(indR) > 0) {
          path.list <- unique(updateList_m(mpath_t, indR, path.list))
        }

      }
    }
  }

  results$LL=LL;
  results$mix_mat=mix_mat;
  return(results)
}
