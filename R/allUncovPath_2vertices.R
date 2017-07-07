allUncovPath_2vertices <- function (p, pag, a, c, suffStat, sepset, mix_modeling, mix_mat)
{

  path.list = c(list(), a);

  Ln=1;
  results=list();
  LL=list();

  while ( (length(path.list) > 0)) {

    cnt=0;
    mpath_t <- path.list[[1]]
    m <- length(mpath_t)
    d <- mpath_t[m]

    path.list[[1]] <- NULL
    visited <- rep(FALSE, p)
    visited[mpath_t] <- TRUE
    uncov <- TRUE

    if (m > 1){
      b_t <- mpath_t[m-1]
      if ( pag[d,c]!=0 & pag[c,d]!=0 & (d %in% sepset[[b_t]][[c]]) & uncov) {

        mpath=c(mpath_t,c)

        n <- length(mpath)

        for (l in seq_len(n-2) ) {

          r = l+2; s = l+1;

          if (pag[mpath[l], mpath[r]] == 0 && pag[mpath[r], mpath[l]] == 0  && mpath[s] %in% sepset[[mpath[l]]][[mpath[r]]]){

            if (mix_mat[mpath[l], mpath[r],mpath[s]]==0 | mix_mat[mpath[r], mpath[l],mpath[s]]==0){
              mix_mat[mpath[l], mpath[r],mpath[s]] = mix_modeling_all(mpath[l], mpath[r], setdiff(sepset[[mpath[l]]][[mpath[r]]],mpath[s]), suffStat, mix_modeling)
              mix_mat[mpath[r], mpath[l],mpath[s]] = mix_mat[mpath[l], mpath[r],mpath[s]];
            }

            if (mix_mat[mpath[l], mpath[r],mpath[s]] == 2) {cnt = cnt + 1;
                if (cnt >1){uncov <- FALSE; break}
            }

          } else {uncov <- FALSE; break}

        }

        if (uncov){

          LL[[Ln]] <- list(path = mpath, cnt = cnt);
          Ln=Ln+1;
        }

      } else {

        indR_t <- which(pag[d,] !=0 & pag[,d] != 0 & !visited)

        indR <- which(sapply(sepset[][[b_t]], function(x) d %in% x))
        #indR = grep(d, sepset[][[b_t]])
        indR = intersect(indR,indR_t);

        if (length(indR) > 0) {
          path.list <- updateList_m(mpath_t, indR, path.list)
        }
      }

    } else if (m==1) {
      if (pag[d,c]!=0 & pag[c,d]!=0 & uncov){

        mpath <- c(d,c);

        LL[[Ln]] <- list(path = mpath, cnt = cnt);
        Ln=Ln+1;
      } else {

        indR <- which(pag[d,] != 0 & pag[,d] !=0 & !visited)

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
