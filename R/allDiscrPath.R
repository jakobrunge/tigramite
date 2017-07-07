#' Finds all discriminating paths for R4

allDiscrPath <- function (pag, start_inter, a, b, c, count, start_stat_edges)
{

  p <- as.numeric(dim(pag)[1])
  visited <- rep(FALSE, p)
  visited[c(a, b, c)] <- TRUE

  indD1 <- which( as.vector(pag[a, ] != 0 & pag[, a] == 2 & !visited ))
  indD2 <- which( as.vector(pag[a, ] != 0 & (pag[, a] == 4 | pag[,a] == 6) & !visited ))

  inter.list=list();
  stat.list=list();

  for (d1 in seq_len(length(indD1))){
    inter.list[[d1]] = start_inter;
    stat.list[[d1]] = start_stat_edges;
  }

  indD2_f = c();
  for (d2 in seq_len(length(indD2))){
    if ( length(intersect( start_inter, count[[indD2[d2]]][[a]][[1]] )) >0){
      indD2_f = c(indD2_f, indD2[d2]);
      inter.list[[length(indD1)+d2]] = intersect( start_inter, count[[indD2[d2]]][[a]][[1]] );
      stat.list[[length(indD1)+d2]] = FALSE;
    }
  }

  indD = c(indD1,indD2_f)

  Ln=1;
  LL=list();

  if (length(indD) > 0) {
    path.list <- updateList_m(a, indD, NULL)

    for (L in 1:length(path.list)){

      while (length(path.list) > 0) {
        mpath <- path.list[[1]]

        interpath <- inter.list[[1]]
        stat_edges <- stat.list[[1]]

        m <- length(mpath)
        d <- mpath[m]

        pred <- mpath[m - 1]

        if ( (pag[c, d] == 0 & pag[d, c] == 0 & pag[d,pred] == 2) |
             (pag[c, d] == 0 & pag[d, c] == 0 & (pag[d,pred] == 4 | pag[d,pred] == 6) & length(intersect(interpath,count[[d]][[pred]][[1]]))>0) )
        {

          if ( pag[d,pred]==4 | pag[d,pred] == 6 ) {stat_edges=FALSE; interpath = intersect(interpath, count[[d]][[pred]][[1]]);}

          LL[[Ln]] <- list(path = c(rev(mpath), b, c), intersect = interpath, stat_edges = stat_edges);
          Ln=Ln+1;

        }

        path.list[[1]] <- NULL
        inter.list[[1]] <- NULL
        stat.list[[1]] <- NULL

        visited <- rep(FALSE, p)
        visited[c(a, b, c)] <- TRUE
        visited[mpath] <- TRUE

        if ( (pag[d,c] == 2 | pag[d,c] == 4 | pag[d,c] == 6) && (pag[c,d] == 3 | pag[c,d] == 5 | pag[c,d] == 6)
             && (pag[pred,d] == 2 | pag[pred, d] == 4 | pag[pred,d] == 6) && (pag[d,pred] == 2 | pag[d,pred]== 4 | pag[d,pred] == 6) ){

          if ( pag[d,pred]==4 | pag[d,pred] == 6 ) {stat_edges = FALSE; interpath = intersect(interpath, count[[d]][[pred]][[1]]);}
          if ( pag[pred,d]==4 | pag[pred,d] == 6 ) {stat_edges = FALSE; interpath = intersect(interpath, count[[pred]][[d]][[1]]);}
          if ( pag[d, c] == 4 | pag[d,c] == 6 ) {stat_edges = FALSE; interpath = intersect(interpath, count[[d]][[c]][[1]]);}
          if ( pag[c, d] == 5 | pag[c,d] == 6 ) {stat_edges = FALSE; interpath = intersect(interpath, count[[c]][[d]][[2]]);}

          indR <- which( as.vector(pag[d, ] != 0 & (pag[,d] == 2 | pag[,d]==4 | pag[,d]==6) &
                                     !visited) )

          if (length(indR) > 0 & length(interpath)>0 ){

            for (r in indR){

              if (pag[r,d]==2){

                path.list <- updateList_m(mpath, r, path.list)
                inter.list = c( inter.list, interpath)
                stat.list = c( stat.list, stat_edges)

              } else if ( (pag[r,d] == 4 | pag[r,d] == 6) & length(intersect(interpath, count[[r]][[d]][[1]]))>0 ){

                path.list <- updateList_m(mpath, r, path.list)
                inter.list = c( inter.list, intersect(interpath, count[[r]][[d]][[1]]) )
                stat.list = c( stat.list, FALSE)

              }
            }
          }
        }
      }
    }
  }
  return(LL)
}
