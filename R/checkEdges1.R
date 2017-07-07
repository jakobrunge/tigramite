#' Checks conditional dependence and mixtures for R4

checkEdges1 <- function (suffStat, indepTest, alpha, apag, sepset, path, mix_modeling, mix_mat)
{
  cnt=0;
  stopifnot((n.path <- length(path)) >= 2)
  found <- FALSE
  SepSet.tot <- unique(c(sepset[[path[1]]][[path[n.path]]],
                         sepset[[path[n.path]]][[path[1]]]))
  if (length(SepSet.tot) != 0) {

    p <- nrow(apag)
    for (i in seq_len(n.path - 1)) {
      x <- path[i]
      y <- path[i + 1]
      SepSet <- setdiff(SepSet.tot, c(x, y))

      if (length(SepSet) != 0) {
        j <- 0
        while (!found && j < length(SepSet)) {
          j <- j + 1
          S.j <- if (j == 1 && length(SepSet) == 1)
            matrix(SepSet, 1, 1)
          else combn(SepSet, j)
          ii <- 0
          while (!found && ii < ncol(S.j)) {
            ii <- ii + 1
            pval = 0;#####
            #pval <- indepTest(x, y, S.j[, ii], suffStat)

            if (pval > alpha) {
              found <- TRUE
              apag[x, y] <- apag[y, x] <- 0
              sepset[[x]][[y]] <- sepset[[y]][[x]] <- S.j[,ii]
            }

            mix_res_j = 1; #####

            #mix_res_j = mix_modeling(x,y,S.j[, ii],suffStat, joint=TRUE);

            #if (mix_res_j == 2){
              #mix_res_s = mix_modeling(x, y, S.j[, ii], suffStat, joint=FALSE);
              #mix_res_s1 = mix_res_s[[1]];
              #mix_res_s2 = mix_res_s[[2]];
            #}


            if (mix_res_j == 1) { cnt=cnt;
            } else if ( mix_res_j ==2 & xor(mix_res_s1 == 1, mix_res_s2 ==1) ) {cnt=cnt+1;
              if (cnt > 1) {found <-TRUE;}
            } else { found <-TRUE; }

          }
        }
      }
    }
  }
  list(deleted = found, apag = apag, sepset = sepset, cnt = cnt, mix_mat = mix_mat)
}
