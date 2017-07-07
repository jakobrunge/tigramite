#' Applies F2CI's orientation rules

OR_discovery2 <- function (apag, suffStat, indepTest, alpha, mix_modeling, sepset, count, rules = rep(TRUE,10), verbose=FALSE)
{

  time_start=proc.time();

  if (any(apag != 0)) {

    p <- ncol(apag)
    mix_mat = array(0,c(p,p,p));
    old_apag <- matrix(0, nrow = p, ncol = p)
    diff_count = 0; max_old_count=0;

    First = TRUE;

    while (any(old_apag != apag) | any(diff_count < max_old_count)) {

      old_apag <- apag
      old_count <- count

      if (rules[1]) {

        ind <- which(   ( (apag == 2 | apag==4 | apag==6) & t(apag) != 0 ), arr.ind = TRUE)
        for (i in seq_len(nrow(ind))) {
          a <- ind[i, 1]
          b <- ind[i, 2]

          indC <- which( apag[,b] != 0 & apag[a, ] == 0 & apag[, a] == 0 & apag[b,] != 0 & apag[,b] != 2) ###
          indC <- setdiff(indC, a)

          if (length(indC) > 0) {

            for (c in indC) {

              if ( (b %in% sepset[[a]][[c]]) ) {

                if (mix_mat[a,c,b]==0 | mix_mat[c,a,b]==0){
                  mix_mat[a,c,b] = mix_modeling_all(a,c,setdiff(sepset[[a]][[c]],b),suffStat,mix_modeling);
                  mix_mat[c,a,b] = mix_mat[a,c,b];
                }

                max_count = max(unlist(count)); if (max_count < 0) {max_count = 0}

                ## begin R1a
                if (apag[a,b] == 2 && mix_mat[a,c,b]==1) {

                  if (apag[c,b] == 1 | apag[c,b] == 5){
                    apag[c, b] <- 3;
                    apag[b,c] <- 2;
                  }

                  if (verbose) {print('rule 1'); print(c(b,c))}

                }
                ## end R1a

                ## begin R1b
                else if (apag[a,b] == 2 & mix_mat[a,c,b]==2 ){

                  if (apag[b,c]!= 0 & apag[b,c] !=2 & apag[b,c] !=3){

                    if (apag[b,c]==5){apag[b, c] <- 6; count[[b]][[c]][[1]] = unique(c(count[[b]][[c]][[1]], max_count+1));
                    } else if (apag[b,c]==4 | apag[b,c]==6){count[[b]][[c]][[1]] = unique(c(count[[b]][[c]][[1]], max_count+1)); ####
                    } else if (apag[b,c]==1) {apag[b, c] <- 4; count[[b]][[c]][[1]] = unique(c(count[[b]][[c]][[1]], max_count+1));
                    }

                  }

                  if (apag[c,b]!= 0 & apag[c,b] !=2 & apag[c,b] !=3){

                    if (apag[c,b]==4){apag[c,b] <- 6; count[[c]][[b]][[2]] = unique(c(count[[c]][[b]][[2]], max_count+1));
                    } else if (apag[c,b]==1) {apag[c,b] <- 5; count[[c]][[b]][[2]] = unique(c(count[[c]][[b]][[2]], max_count+1));
                    } else if (apag[c,b]==5 | apag[c,b]==6) {count[[c]][[b]][[2]] = unique(c(count[[c]][[b]][[2]], max_count+1)); ####
                    }

                  }

                }
                ## end R1b

                ## begin R1c
                else if ( (apag[a,b] == 4 | apag[a,b] == 6 ) & mix_mat[a,c,b]==1 ){

                  if (apag[b,c]!= 0 & apag[b,c] !=2 & apag[b,c] !=3){
                    if (apag[b,c]==5){apag[b, c] <- 6; count[[b]][[c]][[1]] = unique(c(count[[b]][[c]][[1]], count[[a]][[b]][[1]]));
                    } else if (apag[b,c]==4 | apag[b,c]==6){count[[b]][[c]][[1]] = unique(c(count[[b]][[c]][[1]], count[[a]][[b]][[1]]));  ####
                    } else if (apag[b,c] == 1) {apag[b, c] <- 4; count[[b]][[c]][[1]] = unique(c(count[[b]][[c]][[1]], count[[a]][[b]][[1]])); }

                  }

                  if (apag[c,b] ==1 | apag[c,b] == 5){apag[c, b] <- 3;}

                }
                ## end R1c

              }
            }
          }
        }
      }

      if (rules[2]) {
        ind <- which( ( (apag!= 0 & apag != 2 & apag != 3) & t(apag) != 0), arr.ind = TRUE)  ### apag==6 is add-on
        for (i in seq_len(nrow(ind))) {
          a <- ind[i, 1]
          c <- ind[i, 2]
          indB <- which( ((apag[a, ] == 2 | apag[a, ] == 4 | apag[a, ] == 6) & (apag[c,] == 3 | apag[c,] == 5 | apag[c,] == 6)) |
                         ((apag[, a] == 3 | apag[, a] == 5 | apag[, a] == 6) & (apag[,c] == 2 | apag[,c] == 4 | apag[,c] == 6))  )

          if (length(indB) > 0) {

            for (b in indB){

              if ( (apag[a,b]==2 & apag[c,b]==3) | (apag[b,a]==3 & apag[b,c]==2) ){ ## R2a

                apag[a,c]=2;

              } else if ( (apag[a,b]==4 | apag[a,b]==6) & apag[c,b]==3 ){ ## R2b

                if (apag[a,c]==5){apag[a,c]=6; count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], count[[a]][[b]][[1]]));
                } else if (apag[a,c]==1){apag[a,c]=4; count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], count[[a]][[b]][[1]]));
                } else if (apag[a,c]==4 | apag[a,c]==6){count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], count[[a]][[b]][[1]]));  ####
                }

              }   else if ( (apag[b,c]==4 | apag[b,c]==6) & apag[b,a]==3 ){ ## R2b

                if (apag[a,c]==5){apag[a,c]=6; count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], count[[b]][[c]][[1]]));
                } else if (apag[a,c]==1){apag[a,c]=4; count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], count[[b]][[c]][[1]]));
                } else if (apag[a,c]==4 | apag[a,c]==6){count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], count[[b]][[c]][[1]]));  ####
                }

              } else if ( apag[a,b]==2 & (apag[c,b]==5 | apag[c,b] == 6) ){ ## R2c

                if (apag[a,c]==5){apag[a,c]=6; count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], count[[c]][[b]][[2]]));
                } else if (apag[a,c]==1) {apag[a,c]=4; count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], count[[c]][[b]][[2]]));
                } else if (apag[a,c]==4 | apag[a,c]==6){count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], count[[c]][[b]][[2]]));  ####
                }

              }  else if ( apag[b,c]==2 & (apag[b,a]==5 | apag[b,a] == 6) ){ ## R2c

                if (apag[a,c]==5){apag[a,c]=6; count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], count[[b]][[a]][[2]]));
                } else if (apag[a,c]==1) {apag[a,c]=4; count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], count[[b]][[a]][[2]]));
                } else if (apag[a,c]==4 | apag[a,c]==6){count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], count[[b]][[a]][[2]]));  ####
                }

              } else if ( (apag[a,b]==4 | apag[a,b]==6) & (apag[c,b]==5 | apag[c,b] == 6) &
                          length( intersect(count[[a]][[b]][[1]], count[[c]][[b]][[2]]) )>0 ) { ## R2d

                if (apag[a,c]==5){apag[a,c]=6; count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], intersect(count[[a]][[b]][[1]], count[[c]][[b]][[2]]) ));
                } else if (apag[a,c]==1){apag[a,c]=4; count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], intersect(count[[a]][[b]][[1]], count[[c]][[b]][[2]]) ));
                } else if (apag[a,c]==4 | apag[a,c]==6){count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], intersect(count[[a]][[b]][[1]], count[[c]][[b]][[2]]) ));  ####
                }
              } else if ( (apag[b,c]==4 | apag[b,c]==6) & (apag[b,a]==5 | apag[b,a] == 6) &
                          length( intersect(count[[b]][[c]][[1]], count[[b]][[a]][[2]]) )>0 ) { ## R2e

                if (apag[a,c]==5){apag[a,c]=6; count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], intersect(count[[b]][[c]][[1]], count[[b]][[a]][[2]]) ));
                } else if (apag[a,c]==1){apag[a,c]=4; count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], intersect(count[[b]][[c]][[1]], count[[b]][[a]][[2]]) ));
                } else if (apag[a,c]==4 | apag[a,c]==6){count[[a]][[c]][[1]]=unique(c(count[[a]][[c]][[1]], intersect(count[[b]][[c]][[1]], count[[b]][[a]][[2]]) ));  ####
                }
              }

            }
          }
        }
      }

      if (rules[3]) {
        ind <- which(apag != 0 & (t(apag)!= 0 & t(apag) !=2 & t(apag)!=3) , arr.ind = TRUE) ### t(apag)==6 is add-on
        for (i in seq_len(nrow(ind))) {
          b <- ind[i, 1]
          d <- ind[i, 2]
          indAC <- which(apag[b, ] != 0 & (apag[, b] ==
                                             2 | apag[,b]==4 | apag[,b]==6) & apag[, d] != 0 & apag[d, ] != 0)
          if (length(indAC) >= 2) {

            counter <- 0
            while ((counter < (length(indAC) - 1))) {
              counter <- counter + 1
              ii <- counter
              while ((ii < length(indAC))) {
                ii <- ii + 1
                if (apag[indAC[counter], indAC[ii]] == 0 &&
                    apag[indAC[ii], indAC[counter]] == 0) {

                  a = indAC[counter]; c = indAC[ii];

                  if (apag[a,c]==0 & apag[c,a]==0 & d %in% sepset[[a]][[c]]){

                    if (mix_mat[a,c,d]==0 | mix_mat[c,a,d]==0){
                      mix_mat[a,c,d] = mix_modeling_all(a, c, setdiff(sepset[[a]][[c]],d), suffStat, mix_modeling);
                      mix_mat[c,a,d] = mix_mat[a,c,d];
                    }

                    if (mix_mat[a,c,d]==1) {

                      if (apag[a,b] == 2 & apag[c,b] == 2 & (apag[d,b] != 0 & apag[d,b] != 2 & apag[d,b] != 3) ){ ## R3a

                        apag[d, b] <- 2;
                        if (verbose) {print('rule 3'); print(c(d,b))}

                      } else if ( (apag[a,b]==4 | apag[a,b] == 6) & apag[c,b]==2 & (apag[d,b] != 0 & apag[d,b] != 2 & apag[d,b] != 3) ){ ## R3b

                        if (apag[d,b]==1){apag[d, b] <- 4; count[[d]][[b]][[1]] = unique(c(count[[d]][[b]][[1]], count[[a]][[b]][[1]]));
                        } else if (apag[d,b]==5){apag[d,b] <- 6; count[[d]][[b]][[1]] = unique(c(count[[d]][[b]][[1]], count[[a]][[b]][[1]]));
                        } else if (apag[d,b]==4 | apag[d,b]==6){count[[d]][[b]][[1]] = unique(c(count[[d]][[b]][[1]], count[[a]][[b]][[1]]));  ####
                        }

                      } else if ( apag[a,b]==2 & (apag[c,b]==4 | apag[c,b] ==6) & (apag[d,b] != 0 & apag[d,b] != 2 & apag[d,b] != 3) ){ ## R3b

                        if (apag[d,b]==1){apag[d, b] <- 4; count[[d]][[b]][[1]] = unique(c(count[[d]][[b]][[1]], count[[c]][[b]][[1]]));
                        } else if (apag[d,b]==5){apag[d,b] <- 6; count[[d]][[b]][[1]] = unique(c(count[[d]][[b]][[1]], count[[c]][[b]][[1]]));
                        } else if (apag[d,b]==4 | apag[d,b]==6){count[[d]][[b]][[1]] = unique(c(count[[d]][[b]][[1]], count[[c]][[b]][[1]]));  ####
                        }

                      } else if ((apag[a,b] ==4 | apag[a,b] == 6) & (apag[c,b] ==4 | apag[c,b] == 6) & (apag[d,b] != 0 & apag[d,b] != 2 & apag[d,b] != 3)
                                 & length(intersect(count[[a]][[b]][[1]], count[[c]][[b]][[1]]))>0 ){ ## R3c

                        if (apag[d,b]==1){apag[d, b] <- 4; count[[d]][[b]][[1]] = unique(c(count[[d]][[b]][[1]], intersect(count[[a]][[b]][[1]], count[[c]][[b]][[1]])));
                        } else if (apag[d,b]==5){apag[d,b] <- 6; count[[d]][[b]][[1]] = unique(c(count[[d]][[b]][[1]], intersect(count[[a]][[b]][[1]], count[[c]][[b]][[1]])));
                        } else if (apag[d,b]==4 | apag[d,b]==6){count[[d]][[b]][[1]] = unique(c(count[[d]][[b]][[1]], intersect(count[[a]][[b]][[1]], count[[c]][[b]][[1]])));  ####
                        }

                      }

                    } else if (apag[a,b] == 2 & apag[c,b] == 2 & (apag[d,b] != 0 & apag[d,b] != 2 & apag[d,b] != 3) & mix_mat[a,c,d] == 2 ){ ## R3d

                      max_count = max(unlist(count)); if (max_count < 0) {max_count = 0}

                      if (apag[d,b]==1){apag[d, b] <- 4; count[[d]][[b]][[1]] = c(count[[d]][[b]][[1]], max_count+1);
                      } else if (apag[d,b]==5){apag[d,b] <- 6; count[[d]][[b]][[1]] = c(count[[d]][[b]][[1]], max_count+1);
                      } else if (apag[d,b]==4 | apag[d,b]==6){count[[d]][[b]][[1]] = c(count[[d]][[b]][[1]], max_count+1); ####
                      }


                    }
                  }
                }
              }
            }
          }
        }
      }

      if (rules[4]) {
        ind <- which(apag != 0 & t(apag)!=0, arr.ind = TRUE) ### t(apag)==6
        while (length(ind) > 0) {
          b <- ind[1, 1]
          c <- ind[1, 2]
          ind <- ind[-1, , drop = FALSE]
          indA <- which(as.vector(  (apag[b,] == 2 | apag[b,]==4 | apag[b,]==6) & (apag[c,] == 3 | apag[c,]==5 | apag[c,]==6 ) & (apag[,c] == 2 | apag[,c] == 4 | apag[,c]==6) ))

          while (length(indA) > 0) {

            a <- indA[1]
            indA <- indA[-1]
            Done <- FALSE

            while (!Done) {

              stat_edges = TRUE;
              interABC = seq_len(max(0,max(unlist(count))));

              if (apag[b,a]==4 | apag[b,a] == 6) {stat_edges=FALSE; interABC = intersect(interABC, count[[b]][[a]][[1]]);}

              if (apag[a,c]==4 | apag[a,c] == 6) {stat_edges=FALSE; interABC = intersect(interABC, count[[a]][[c]][[1]]);}

              if (apag[c,a]==5 | apag[c,a] == 6) {stat_edges=FALSE; interABC = intersect(interABC, count[[c]][[a]][[2]]);}

              if ( stat_edges == FALSE & length(interABC)==0 ){
                Done <- TRUE;
              }

              if (Done == FALSE){
                md.path1 <- allDiscrPath(apag, interABC, a, b, c, count, stat_edges)
              } else {md.path1 = NULL;
              }

              if (length(md.path1)==0) {
                Done <- TRUE
              } else {

                for (p1 in seq_len(length(md.path1))){

                  md.path = as.vector(md.path1[[p1]]$path);

                  interAll = md.path1[[p1]]$intersect;   ##!!!
                  stat_edges= md.path1[[p1]]$stat_edges;  ##!!!

                  N.md <- length(md.path);

                  chkE <- checkEdges1(suffStat, indepTest,
                                      alpha = alpha, apag = apag, sepset = sepset,
                                      path = md.path, mix_modeling, mix_mat)
                  sepset <- chkE$sepset
                  apag <- chkE$apag
                  cnt <- chkE$cnt
                  mix_mat <- chkE$mix_mat

                  if ( !chkE$deleted & ((stat_edges==TRUE & cnt==0)|(stat_edges==TRUE & cnt==1)|(stat_edges==FALSE & cnt==0)) ) { ###

                    if (b %in% sepset[[md.path[1]]][[md.path[N.md]]] ||
                        b %in% sepset[[md.path[N.md]]][[md.path[1]]]) {

                      if (stat_edges==TRUE & cnt==0){ ## R4a, part 1
                        if (apag[b,c]==1 | apag[b,c] == 4) apag[b, c] <- 2;
                        if (apag[c,b]==1 | apag[c,b] == 5) apag[c,b] <- 3;
                        ## already have apag[c,b] != 2 & apag[c,b] != 3
                        if (verbose) {print('rule 4'); print(c(b,c))}

                      } else if (stat_edges==FALSE & cnt == 0){ ## R4c, part 1
                        if ( apag[c,b]!=3 & apag[c,b]!=6 & length(interAll)>0) {
                          if (apag[b,c] ==1 | apag[b,c] == 4) {apag[b,c] = 4; count[[b]][[c]][[1]] = unique(c( count[[b]][[c]][[1]], interAll)); ###
                          } else if (apag[b,c] ==5 | apag[b,c] == 6) {apag[b,c] = 6; count[[b]][[c]][[1]] = unique(c( count[[b]][[c]][[1]], interAll)); ###
                          }
                          if (apag[c,b]==1 | apag[c,b] == 5) {apag[c,b] = 5; count[[c]][[b]][[2]] = unique(c( count[[c]][[b]][[2]], interAll));
                          } else if (apag[c,b]==4 | apag[c,b] == 6) {apag[c,b] = 6; count[[c]][[b]][[2]] = unique(c( count[[c]][[b]][[2]], interAll));
                          }
                        }

                      } else if (stat_edges==TRUE & cnt == 1){ ## R4b, part 1
                        max_count = max(unlist(count)); if (max_count < 0) {max_count = 0}
                        if (apag[b,c] == 1 | apag[b,c] == 4) {apag[b,c] = 4; count[[b]][[c]][[1]] = c( count[[b]][[c]][[1]], max_count + 1);###
                        } else if (apag[b,c]==5 | apag[b,c]==6) {apag[b,c] = 6; count[[b]][[c]][[1]] = c( count[[b]][[c]][[1]], max_count + 1); ###
                        }
                        if (apag[c,b] == 1 | apag[c,b]==5) apag[c,b] = 3; ###
                      }
                    }

                    else {

                      if (stat_edges==TRUE & cnt == 0){ ## R4a, part 2
                        if (apag[a,b]==1 | apag[a,b]==4) apag[a,b] = 2;
                        if (apag[c,b]==1 | apag[c,b]==4) apag[c,b] = 2;
                        if (apag[b,c]==1 | apag[b,c]==4) apag[b,c] = 2;
                        if (verbose) print('rule 4')

                      } else if (stat_edges==FALSE & cnt == 0){ ## R4c, part 2
                        if ((apag[c,b] == 1 | apag[c,b]==4) & length(interAll)>0) {
                          apag[c,b] = 4;
                          count[[c]][[b]][[1]] = unique(c( count[[c]][[b]][[1]], interAll));
                        } else if ((apag[c,b] == 5 | apag[c,b] == 6) & length(interAll)>0) {
                          apag[c,b] = 6;
                          count[[c]][[b]][[1]] = unique(c( count[[c]][[b]][[1]], interAll));
                        }
                        if ((apag[a,b] == 5 | apag[a,b]==6) & length(interAll)>0) {
                          apag[a,b] = 6;
                          count[[a]][[b]][[1]] = unique(c( count[[a]][[b]][[1]], interAll));
                        } else if ((apag[a,b] == 1 | apag[a,b]== 4) & length(interAll)>0){
                          apag[a,b] = 4;
                          count[[a]][[b]][[1]] = unique(c( count[[a]][[b]][[1]], interAll));
                        }
                        if ((apag[b,c] == 5 | apag[b,c]==6) & length(interAll)>0) {
                          apag[b,c] = 6;
                          count[[b]][[c]][[1]] = unique(c( count[[b]][[c]][[1]], interAll));
                        } else if ((apag[b,c] == 1 | apag[b,c]== 4) & length(interAll)>0){
                          apag[b,c] = 4;
                          count[[b]][[c]][[1]] = unique(c( count[[b]][[c]][[1]], interAll));
                        }

                      } else if (stat_edges==TRUE & cnt == 1){  ## R4b, part 2
                        max_count = max(unlist(count)); if (max_count < 0) {max_count = 0}
                        if (apag[c,b] == 1 | apag[c,b]==4) {
                          apag[c,b] = 4;
                          count[[c]][[b]][[1]] = c( count[[c]][[b]][[1]], max_count + 1);
                        } else if (apag[c,b] == 5 | apag[c,b]==6) {
                          apag[c,b] = 6;
                          count[[c]][[b]][[1]] = c( count[[c]][[b]][[1]], max_count + 1);
                        }
                        if (apag[a,b] == 5 | apag[a,b]==6) {
                          apag[a,b] = 6;
                          count[[a]][[b]][[1]] = c( count[[a]][[b]][[1]], max_count + 1);
                        } else if (apag[a,b] == 1 | apag[a,b]== 4){
                          apag[a,b] = 4;
                          count[[a]][[b]][[1]] = c( count[[a]][[b]][[1]], max_count + 1);
                        }
                        if (apag[b,c] == 5 | apag[b,c]==6) {
                          apag[b,c] = 6;
                          count[[b]][[c]][[1]] = c( count[[b]][[c]][[1]], max_count + 1);
                        } else if (apag[b,c] == 1 | apag[b,c]== 4){
                          apag[b,c] = 4;
                          count[[b]][[c]][[1]] = c( count[[b]][[c]][[1]], max_count + 1);
                        }
                      }

                    }
                  }
                  if (p1==length(md.path1)){Done <- TRUE;}
                }
              }
            }
          }
        }
      }

      # if (rules[5]) {
      #
      #   ind <- which( (apag != 0 & apag != 2 & apag != 3) & (t(apag)!=0 & t(apag)!=2 & t(apag)!=3), arr.ind = TRUE) ### apag==6, t(apag)==6
      #   ucp.path = list();
      #   while (length(ind) > 0) {
      #     a <- ind[1, 1]
      #     b <- ind[1, 2]
      #     ind <- ind[-1, , drop = FALSE]
      #     indC <- which( (apag[a, ] !=0 & apag[a, ] !=2 & apag[a,] !=3) & (apag[,a] != 0 & apag[,a] != 2 & apag[,a] !=3) & (apag[b, ] == 0 & apag[, b] == 0)) ### apag[a,]==6, apag[,a]==6
      #     indC <- setdiff(indC, b)
      #
      #     indD <- which( (apag[b, ] !=0 & apag[b, ] !=2 & apag[b,] !=3) & (apag[,b] !=0 & apag[,b] !=2 & apag[,b]!=3) & (apag[a, ] == 0 & apag[, a] == 0) ) ### apag[b,]==6, apag[,b]==6
      #     indD <- setdiff(indD, a)
      #
      #
      #     if (length(indC) > 0 && length(indD) > 0) {
      #       counterC <- 0
      #       while ( counterC < length(indC) ) {
      #         counterC <- counterC + 1
      #         c <- indC[counterC]
      #
      #         counterD <- 0
      #         while ( counterD < length(indD) ) {
      #           counterD <- counterD + 1
      #           d <- indD[counterD]
      #
      #           ucp_t <- allUncovCircPath(p, pag = apag, path = c(a, c, d, b), sepset, suffStat, count, mix_modeling, mix_mat)
      #           mix_mat = ucp_t$mix_mat;
      #           ucp.path = c(ucp.path, ucp_t$LL);
      #
      #         }
      #       }
      #     }
      #   }
      #
      #   for (p1 in seq_len(length(ucp.path))) {
      #
      #     ucp = ucp.path[[p1]]$path;
      #     cnt = ucp.path[[p1]]$cnt;
      #
      #     if (length(ucp) > 1) {
      #       n <- length(ucp)
      #
      #       if (cnt==0) {
      #
      #         for (j in seq_len(n)) {
      #
      #           if (j==1){
      #             apag[ucp[n], ucp[1]] <- 3;
      #             apag[ucp[2], ucp[1]] <- 3;
      #           } else if (j==n){
      #             apag[ucp[1], ucp[n]] <- 3;
      #             apag[ucp[n-1], ucp[n]] <- 3;
      #           } else {
      #             apag[ucp[j-1], ucp[j]]<-3;
      #             apag[ucp[j+1], ucp[j]]<-3;
      #           }
      #
      #           if (verbose) print('rule 5')
      #
      #         }
      #       } else if (cnt>0) {
      #
      #         max_count = max(unlist(count)); if (max_count < 0) {max_count = 0}
      #
      #         for (j in seq_len(n)) {
      #
      #           if (j==1){
      #             if (apag[ucp[n], ucp[1]]==1 | apag[ucp[n], ucp[1]]==5) {apag[ucp[n], ucp[1]]=5; count[[ucp[n]]][[ucp[1]]][[2]] = c( count[[ucp[n]]][[ucp[1]]][[2]], max_count+1 ) ;}
      #             if (apag[ucp[n], ucp[1]]==4 | apag[ucp[n], ucp[1]]==6) {apag[ucp[n], ucp[1]]=6; count[[ucp[n]]][[ucp[1]]][[2]] = c( count[[ucp[n]]][[ucp[1]]][[2]], max_count+1 ) ;}
      #
      #             if (apag[ucp[2], ucp[1]]==1 | apag[ucp[2], ucp[1]]==5) {apag[ucp[2], ucp[1]]=5; count[[ucp[2]]][[ucp[1]]][[2]] = c( count[[ucp[2]]][[ucp[1]]][[2]], max_count+1 ) ;}
      #             if (apag[ucp[2], ucp[1]]==4 | apag[ucp[2], ucp[1]]==6) {apag[ucp[2], ucp[1]]=6; count[[ucp[2]]][[ucp[1]]][[2]] = c( count[[ucp[2]]][[ucp[1]]][[2]], max_count+1 ) ;}
      #
      #           } else if (j==n){
      #             if (apag[ucp[1], ucp[n]]==1 | apag[ucp[1], ucp[n]]==5) {apag[ucp[1], ucp[n]]=5; count[[ucp[1]]][[ucp[n]]][[2]] = c( count[[ucp[1]]][[ucp[n]]][[2]], max_count+1 ) ;}
      #             if (apag[ucp[1], ucp[n]]==4 | apag[ucp[1], ucp[n]]==6) {apag[ucp[1], ucp[n]]=6; count[[ucp[1]]][[ucp[n]]][[2]] = c( count[[ucp[1]]][[ucp[n]]][[2]], max_count+1 ) ;}
      #
      #             if (apag[ucp[n-1], ucp[n]]==1 | apag[ucp[n-1], ucp[n]]==5) {apag[ucp[n-1], ucp[n]]=5; count[[ucp[n-1]]][[ucp[n]]][[2]] = c( count[[ucp[n-1]]][[ucp[n]]][[2]], max_count+1 ) ;}
      #             if (apag[ucp[n-1], ucp[n]]==4 | apag[ucp[n-1], ucp[n]]==6) {apag[ucp[n-1], ucp[n]]=6; count[[ucp[n-1]]][[ucp[n]]][[2]] = c( count[[ucp[n-1]]][[ucp[n]]][[2]], max_count+1 ) ;}
      #
      #           } else {
      #
      #             if (apag[ucp[j+1], ucp[j]]==1 | apag[ucp[j+1], ucp[j]]==5) {apag[ucp[j+1], ucp[j]]=5; count[[ucp[j+1]]][[ucp[j]]][[2]] = c( count[[ucp[j+1]]][[ucp[j]]][[2]], max_count+1 ) ;}
      #             if (apag[ucp[j+1], ucp[j]]==4 | apag[ucp[j+1], ucp[j]]==6) {apag[ucp[j+1], ucp[j]]=6; count[[ucp[j+1]]][[ucp[j]]][[2]] = c( count[[ucp[j+1]]][[ucp[j]]][[2]], max_count+1 ) ;}
      #
      #             if (apag[ucp[j-1], ucp[j]]==1 | apag[ucp[j-1], ucp[j]]==5) {apag[ucp[j-1], ucp[j]]=5; count[[ucp[j-1]]][[ucp[j]]][[2]] = c( count[[ucp[j-1]]][[ucp[j]]][[2]], max_count+1 ) ;}
      #             if (apag[ucp[j-1], ucp[j]]==4 | apag[ucp[j-1], ucp[j]]==6) {apag[ucp[j-1], ucp[j]]=6; count[[ucp[j-1]]][[ucp[j]]][[2]] = c( count[[ucp[j-1]]][[ucp[j]]][[2]], max_count+1 ) ;}
      #
      #           }
      #
      #         }
      #
      #       }
      #
      #     }
      #   }
      # }

      if (rules[6]) {
        ind <- which((apag != 0 & (t(apag)!= 0 & t(apag) !=2 & t(apag) != 3) ), arr.ind = TRUE)
        for (i in seq_len(nrow(ind))) {
          b <- ind[i, 1]
          c <- ind[i, 2]

          indA <- which( (apag[b,]==3 | apag[b,]==5 | apag[b,]==6) & (apag[,b]==3 | apag[,b]==5 | apag[,b]==6) )
          if (length(indA) > 0) {

            for (a in indA){

              if (apag[a,b]==3 & apag[b,a]==3 & (apag[c,b]== 1 | apag[c,b] == 5) ){ ## R6a

                apag[c,b]<-3
                if (verbose) {print('rule 6');print(c(c,b))}

              } else if ( (apag[a,b]==5|apag[a,b]==6) && apag[b,a]==3 ){  ## R6b

                if (apag[c,b] == 1 | apag[c,b] == 5) {apag[c,b]<-5; count[[c]][[b]][[2]] = unique(c( count[[c]][[b]][[2]], count[[a]][[b]][[2]] ));
                } else if (apag[c,b] == 4 | apag[c,b]==6) {apag[c,b]<-6; count[[c]][[b]][[2]] = unique(c( count[[c]][[b]][[2]], count[[a]][[b]][[2]] ));}

              } else if (apag[a,b]==3 && (apag[b,a]==5|apag[b,a]==6) ){  ## R6c

                if (apag[c,b] == 1 | apag[c,b] == 5) {apag[c,b]<-5; count[[c]][[b]][[2]] = unique(c( count[[c]][[b]][[2]], count[[b]][[a]][[2]] ));
                } else if (apag[c,b] == 4 | apag[c,b]==6) {apag[c,b]<-6; count[[c]][[b]][[2]] = unique(c( count[[c]][[b]][[2]], count[[b]][[a]][[2]] ));}

              } else if ( (apag[a,b]==5|apag[a,b]==6) && (apag[b,a]==5|apag[b,a]==6) && length(intersect(count[[a]][[b]][[2]], count[[b]][[a]][[2]]))>0 ){  ## R6d

                if (apag[c,b] == 1 | apag[c,b] == 5) {apag[c,b]<-5; count[[c]][[b]][[2]] = unique(c( count[[c]][[b]][[2]], intersect(count[[a]][[b]][[2]], count[[b]][[a]][[2]]) ));
                } else if (apag[c,b] == 4 | apag[c,b]==6) {apag[c,b]<-6; count[[c]][[b]][[2]] = unique(c( count[[c]][[b]][[2]], intersect(count[[a]][[b]][[2]], count[[b]][[a]][[2]]) ));}
              }
            }

          }
        }
      }

      if (rules[7]) {
        ind <- which((apag != 0 & (t(apag)!= 0 & t(apag) != 2 & t(apag) != 3)), arr.ind = TRUE)
        for (i in seq_len(nrow(ind))) {
          b <- ind[i, 1]
          c <- ind[i, 2]

          indA <- which( ( (apag[b, ] == 3 | apag[b, ] == 5 | apag[b,]==6) & apag[, b] !=0
          ) & (apag[c, ] == 0 & apag[, c] == 0) )
          indA <- setdiff(indA, c)

          for (a in indA){
            if (b %in% sepset[[a]][[c]]) {

              if (mix_mat[a,c,b]==0 | mix_mat[c,a,b]==0){
                mix_mat[a,c,b] = mix_modeling_all(a, c, setdiff(sepset[[a]][[c]],a), suffStat, mix_modeling);
                mix_mat[a,c,b] = mix_mat[c,a,b];
              }

              max_count = max(unlist(count)); if (max_count < 0) {max_count = 0}

              if (apag[b,a]==3 & mix_mat[a,c,b] ==1 & (apag[c,b]== 1 | apag[c,b] == 5)) { ## R7a

                apag[c, b] <- 3;
                if (verbose) {print('rule 7'); print(c(c,b))}

              } else if ( (apag[b,a]==5|apag[b,a]==6) & mix_mat[a,c,b] ==1 ) { ## R7b

                if (apag[c,b] == 1 | apag[c,b] == 5) {apag[c, b] <- 5; count[[c]][[b]][[2]] = unique(c( count[[c]][[b]][[2]], count[[b]][[a]][[2]] ));
                } else if (apag[c,b] == 4 | apag[c,b] == 6) {apag[c,b] <- 6; count[[c]][[b]][[2]] = unique(c( count[[c]][[b]][[2]], count[[b]][[a]][[2]] ));}


              } else if (apag[b,a]==3 &  mix_mat[a,c,b] ==2 ) { ## R7c

                if (apag[c,b] == 1 | apag[c,b] == 5) {apag[c, b] <- 5; count[[c]][[b]][[2]] = c( count[[c]][[b]][[2]], max_count+1 );
                } else if (apag[c,b] == 4 | apag[c,b] == 6) {apag[c,b] <- 6; count[[c]][[b]][[2]] = c( count[[c]][[b]][[2]], max_count+1 );}

              }
            }
          }
        }
      }

      if (rules[8]) {
        ind <- which( (apag != 0 & (t(apag)!= 0 & t(apag) != 2 & t(apag) != 3)), arr.ind = TRUE)
        for (i in seq_len(nrow(ind))) {
          a <- ind[i, 1]
          c <- ind[i, 2]
          indB <- which( apag[a, ] !=0 & (apag[, a] == 3 | apag[,a] == 5 | apag[,a] == 6) & (apag[c,] == 3 | apag[c,] == 5 | apag[c,] == 6) & apag[,c] != 0)
          if (length(indB) > 0) {

            for (b in indB){

              if (apag[b,a]==3 && apag[c,b]==3 & (apag[c,a]== 1 | apag[c,a] == 5)){ ## R8a

                apag[c,a]<- 3;
                if (verbose) print('rule 8')

              } else if ( apag[b,a]==3 && (apag[c,b]==5 | apag[c,b]==6) ){ ## R8c

                if (apag[c,a]==1 | apag[c,a]==5){apag[c,a]=5; count[[c]][[a]][[2]] = unique(c( count[[c]][[a]][[2]], count[[c]][[b]][[2]] ));
                } else if (apag[c,a]==4 | apag[c,a]==6){apag[c,a]=6; count[[c]][[a]][[2]] = unique(c( count[[c]][[a]][[2]], count[[c]][[b]][[2]] ));}

              } else if ( (apag[b,a]==5 | apag[b,a]==6) && apag[c,b]==3 ) { ## R8b

                if (apag[c,a]==1 | apag[c,a]==5){apag[c,a]=5; count[[c]][[a]][[2]] = unique(c( count[[c]][[a]][[2]], count[[b]][[a]][[2]] ));
                } else if (apag[c,a]==4 | apag[c,a]==6){apag[c,a]=6; count[[c]][[a]][[2]] = unique(c( count[[c]][[a]][[2]], count[[b]][[a]][[2]] ));}


              } else if ( (apag[b,a]==5 | apag[b,a]==6) && (apag[c,b]==5 | apag[c,b]==6) &
                          length(setdiff(intersect(count[[b]][[a]][[2]], count[[c]][[b]][[2]]),count[[c]][[a]][[2]]))>0) {  ## R8d

                if (apag[c,a]==1 | apag[c,a]==5){apag[c,a]=5; count[[c]][[a]][[2]] = unique(c( count[[c]][[a]][[2]], intersect(count[[b]][[a]][[2]], count[[c]][[b]][[2]]) ));
                } else if (apag[c,a]==4 | apag[c,a]==6){apag[c,a]=6; count[[c]][[a]][[2]] = unique(c( count[[c]][[a]][[2]], intersect(count[[b]][[a]][[2]], count[[c]][[b]][[2]]) ));}


              }
            }

          }
        }
      }

      if (rules[9]) {
        ind <- which(apag != 0 & (t(apag) != 0 & t(apag) !=2 & t(apag) != 3) , arr.ind = TRUE)
        upd = list();
        while (length(ind) > 0) {
          a <- ind[1, 1]
          c <- ind[1, 2]

          ind <- ind[-1, , drop = FALSE]

          indB <- which(sapply(sepset[][[c]], function(x) a %in% x))
          #indB <- grep(a, sepset[][[c]])

          indB <- setdiff(indB, c)

          while ((length(indB) > 0)) {
            b <- indB[1]
            indB <- indB[-1]

            if (a %in% sepset[[b]][[c]]){

              upd_t <- allUncovPath(p, apag, a, b, c, suffStat, sepset, mix_modeling, mix_mat)
              mix_mat = upd_t$mix_mat;
              upd = c(upd, upd_t$LL);
            }
          }
        }

        for (p1 in seq_len(length(upd))) {

          npath = upd[[p1]]$path

          a = npath[1];
          b = npath[2];
          c = npath[length(npath)];

          cnt = upd[[p1]]$cnt;

          if (mix_mat[b,c,a]==0 | mix_mat[c,b,a]==0){
            mix_mat[b,c,a] = mix_modeling_all(b, c, setdiff(sepset[[b]][[c]],a), suffStat, mix_modeling);
            mix_mat[c,b,a] = mix_mat[b,c,a];
          }

          if ( mix_mat[b,c,a] == 2) {cnt = cnt + 1;}

          if (length(npath) > 1){
            if (cnt == 0 & (apag[c,a] ==1 | apag[c,a] == 5) ) { ## R9a
              apag[c, a] <- 3
              if (verbose) {print('rule 9'); print(c(c,a))}
            } else if (cnt==1) {  ## R9b and R9c
              max_count = max(unlist(count)); if (max_count < 0) {max_count = 0}
              if (apag[c,a] == 1 | apag[c,a]==5) {apag[c,a] = 5; count[[c]][[a]][[2]] = c( count[[c]][[a]][[2]], max_count+1 );
              } else if (apag[c,a] == 4 | apag[c,a]==6) {apag[c,a] = 6; count[[c]][[a]][[2]] = c( count[[c]][[a]][[2]], max_count+1 );}

            }
          }
        }
      }

      if (rules[10]) {
        ind <- which( (t(apag)!=0 & t(apag)!=2 & t(apag)!=3 & apag != 0), arr.ind = TRUE)
        while (length(ind) > 0) {
          a <- ind[1, 1]
          c <- ind[1, 2]
          ind <- ind[-1, , drop = FALSE]
          indB <- which((apag[c, ] == 3 | apag[c, ] == 5 | apag[c, ] == 6) & apag[,c] != 0)
          if (length(indB) >= 2) {
            counterB <- 0
            while (counterB < length(indB)) {
              counterB <- counterB + 1
              b <- indB[counterB]
              indD <- setdiff(indB, b)
              counterD <- 0
              while ((counterD < length(indD))) {
                counterD <- counterD + 1
                d <- indD[counterD]

                atmp1 <- allUncovPath_2vertices(p, apag, a, b, suffStat, sepset, mix_modeling, mix_mat)
                mix_mat = atmp1$mix_mat;
                atmp2 <- allUncovPath_2vertices(p, apag, a, d, suffStat, sepset, mix_modeling, mix_mat)
                mix_mat = atmp2$mix_mat;

                atmp1=atmp1$LL;
                atmp2=atmp2$LL;

                for (t1 in seq_len(length(atmp1))) {
                  for (t2 in seq_len(length(atmp2))) {

                    tmp1 = atmp1[[t1]];
                    tmp2 = atmp2[[t2]];

                    if (length(tmp1$path) > 1 && length(tmp2$path) >
                        1 && (tmp1$path[2] != tmp2$path[2]) && apag[tmp1$path[2],tmp2$path[2]] == 0 && apag[tmp2$path[2],tmp1$path[2]] == 0 && (a %in% sepset[[tmp1$path[2]]][[tmp2$path[2]]] ) ) {

                      if ( mix_mat[tmp1$path[2],tmp2$path[2],a]==0 | mix_mat[tmp2$path[2],tmp1$path[2],a]==0){
                        mix_mat[tmp1$path[2],tmp2$path[2],a] = mix_modeling_all(tmp1$path[2],tmp2$path[2],setdiff(sepset[[tmp1$path[2]]][[tmp2$path[2]]],a),suffStat,mix_modeling);
                        mix_mat[tmp2$path[2],tmp1$path[2],a] = mix_mat[tmp1$path[2],tmp2$path[2],a];
                      }

                      if ( (apag[c,a] == 1 | apag[c,a] == 5) & apag[c,b] == 3 & apag[c,d] == 3 & (tmp1$cnt == 0 & tmp2$cnt == 0) & mix_mat[tmp1$path[2],tmp2$path[2],a] == 1) { ## R10a

                        apag[c, a] <- 3;
                        if (verbose) print('rule 10')

                      } else if ( (apag[c,a] == 1 | apag[c,a] == 4) & ((apag[c,b] == 5 | apag[c,b] == 6) & apag[c,d] == 3) & tmp1$cnt == 0 & tmp2$cnt == 0 &
                                  mix_mat[tmp1$path[2],tmp2$path[2],a] == 1){ ## R10b

                        if (apag[c,a] == 1 | apag[c,a] == 5) {apag[c,a] <- 5; count[[c]][[a]][[2]] = unique(c( count[[c]][[a]][[2]], count[[c]][[b]][[2]] ));
                        } else if (apag[c,a] == 4 | apag[c,a] == 6) {apag[c,a] <- 6; count[[c]][[a]][[2]] = unique(c( count[[c]][[a]][[2]], count[[c]][[b]][[2]] ));
                        }

                      } else if ( (apag[c,a] == 1 | apag[c,a] == 4) & (apag[c,b] == 3 & (apag[c,d] == 5|apag[c,d] == 6)) & tmp1$cnt == 0 & tmp2$cnt == 0 &
                                  mix_mat[tmp1$path[2],tmp2$path[2],a] == 1 ){ ## R10b

                        if (apag[c,a] == 1 | apag[c,a] == 5) {apag[c,a] <- 5; count[[c]][[a]][[2]] = unique(c( count[[c]][[a]][[2]], count[[c]][[d]][[2]] )); fin = 1;
                        } else if (apag[c,a] == 4 | apag[c,a] == 6) {apag[c,a] <- 6; count[[c]][[a]][[2]] = unique(c( count[[c]][[a]][[2]], count[[c]][[d]][[2]] ));fin = 1;
                        }

                      }else if ( (apag[c,a] == 1 | apag[c,a] == 4) & apag[c,b] == 3 & apag[c,d] == 3 & xor(tmp1$cnt==1,tmp2$cnt == 1) & mix_mat[tmp1$path[2],tmp2$path[2],a] == 1 ) {  ## R10d

                        max_count = max(unlist(count)); if (max_count < 0) {max_count = 0}

                        if (apag[c,a] == 1 | apag[c,a] == 5) {apag[c,a] <- 5; count[[c]][[a]][[2]] = c( count[[c]][[a]][[2]], max_count+1 );
                        } else if (apag[c,a] == 4 | apag[c,a] == 6) {apag[c,a] <- 6; count[[c]][[a]][[2]] = c( count[[c]][[a]][[2]], max_count+1 );
                        }

                      } else if ( (apag[c,a] == 1 | apag[c,a] == 4) & apag[c,b] == 3 & apag[c,d] == 3 & tmp1$cnt == 0 & tmp2$cnt == 0 & mix_mat[tmp1$path[2],tmp2$path[2],a] == 2 ) {  ## R10e

                        max_count = max(unlist(count)); if (max_count < 0) {max_count = 0}

                        if (apag[c,a] == 1 | apag[c,a] == 5) {apag[c,a] <- 5; count[[c]][[a]][[2]] = c( count[[c]][[a]][[2]], max_count+1 );
                        } else if (apag[c,a] == 4 | apag[c,a] == 6) {apag[c,a] <- 6; count[[c]][[a]][[2]] = c( count[[c]][[a]][[2]], max_count+1 );
                        }

                      } else if ( (apag[c,a] == 1 | apag[c,a] == 4) & ((apag[c,b] == 5 | apag[c,b] == 6) & (apag[c,d] == 5|apag[c,d] == 6))
                                  & length(intersect(count[[c]][[b]][[2]],count[[c]][[d]][[2]]))>0 & tmp1$cnt == 0 & tmp2$cnt == 0 & mix_mat[tmp1$path[2],tmp2$path[2],a] == 1) {  ## R10c

                        if (apag[c,a] == 1 | apag[c,a] == 5) {apag[c,a] <- 5; count[[c]][[a]][[2]] = unique(c( count[[c]][[a]][[2]], intersect(count[[c]][[b]][[2]],count[[c]][[d]][[2]]) ));
                        } else if (apag[c,a] == 4 | apag[c,a] == 6) {apag[c,a] <- 6; count[[c]][[a]][[2]] = unique(c( count[[c]][[a]][[2]], intersect(count[[c]][[b]][[2]],count[[c]][[d]][[2]]) ));
                        }

                      }
                    }
                  }
                }
              }
            }
          }
        }
      }


      diff_count <- unlist(lapply(1:length(count),function(i)
        lapply(1:length(count[[i]]), function (j)
          lapply(1:2, function(k) setdiff(count[[i]][[j]][[k]][count[[i]][[j]][[k]]>0], old_count[[i]][[j]][[k]][old_count[[i]][[j]][[k]]>0])))));

      if (length(diff_count)==0){diff_count = Inf};

      if (First & all(old_apag == apag)){
        max_old_count <- max(unlist(old_count));
        First = FALSE;}
    }
  }

  time_or = proc.time() - time_start;

  OR_results = list(G=apag, time_or = time_or);

  return(OR_results)
}
