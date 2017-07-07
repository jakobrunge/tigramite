#' Find triples with dependent edges for RFCI

dep.triple <- function (suffStat, indepTest, alpha, sepset, apag, unshTripl,
          unshVect, trueVstruct, verbose = FALSE)
{
  p <- nrow(apag)
  for (k in seq_len(ncol(unshTripl))) {
    if (trueVstruct[k]) {
      x <- unshTripl[1, k]
      y <- unshTripl[2, k]
      z <- unshTripl[3, k]
      SepSet <- setdiff(unique(c(sepset[[x]][[z]], sepset[[z]][[x]])),
                        y)
      nSep <- length(SepSet)
      if (verbose)
        cat("\nTriple:", x, y, z, "and sepSet (of size",
            nSep, ") ", SepSet, "\n")
      if (nSep != 0) {
        x. <- min(x, y)
        y. <- max(x, y)
        y_ <- min(y, z)
        z_ <- max(y, z)
        del1 <- FALSE
        if (indepTest(x, y, SepSet, suffStat) >= alpha) {
          del1 <- TRUE
          done <- FALSE
          ord <- 0L
          while (!done && ord < nSep) {
            ord <- ord + 1L
            S.j <- if (ord == 1 && nSep == 1)
              matrix(SepSet, 1, 1)
            else combn(SepSet, ord)
            for (i in seq_len(ncol(S.j))) {
              pval <- indepTest(x, y, S.j[, i], suffStat)
              if (verbose)
                cat("x=", x, " y=", y, "S=", S.j[, i],
                    ": pval =", pval, "\n")
              if (pval >= alpha) {
                apag[x, y] <- apag[y, x] <- 0
                sepset[[x]][[y]] <- sepset[[y]][[x]] <- S.j[,
                                                            i]
                done <- TRUE
                break
              }
            }
          }
          indM <- which((apag[x, ] == 1 & apag[, x] ==
                           1) & (apag[y, ] == 1 & apag[, y] == 1))
          indM <- setdiff(indM, c(x, y, z))
          for (m in indM) {
            unshTripl <- cbind(unshTripl, c(x., m, y.))
            unshVect <- c(unshVect, triple2numb(p, x.,
                                                m, y.))
            trueVstruct <- c(trueVstruct, TRUE)
          }
          indQ <- which((apag[x, ] == 1 & apag[, x] ==
                           1) & (apag[y, ] == 0 & apag[, y] == 0))
          indQ <- setdiff(indQ, c(x, y, z))
          for (q in indQ) {
            delTripl <- unshVect == (if (q < y)
              triple2numb(p, q, x, y)
              else triple2numb(p, y, x, q))
            if (any(delTripl))
              trueVstruct[which.max(delTripl)] <- FALSE
          }
          indR <- which((apag[x, ] == 0 & apag[, x] ==
                           0) & (apag[y, ] == 1 & apag[, y] == 1))
          indR <- setdiff(indR, c(x, y, z))
          for (r in indR) {
            delTripl <- unshVect == (if (r < x)
              triple2numb(p, r, y, x)
              else triple2numb(p, x, y, r))
            if (any(delTripl))
              trueVstruct[which.max(delTripl)] <- FALSE
          }
        }
        del2 <- FALSE
        if (indepTest(z, y, SepSet, suffStat) >= alpha) {
          del2 <- TRUE
          Done <- FALSE
          Ord <- 0L
          while (!Done && Ord < nSep) {
            Ord <- Ord + 1L
            S.j <- if (Ord == 1 && nSep == 1)
              matrix(SepSet, 1, 1)
            else combn(SepSet, Ord)
            for (i in seq_len(ncol(S.j))) {
              pval <- indepTest(z, y, S.j[, i], suffStat)
              if (verbose)
                cat("x=", z, " y=", y, " S=", S.j[, i],
                    ": pval =", pval, "\n")
              if (pval >= alpha) {
                apag[z, y] <- apag[y, z] <- 0
                sepset[[z]][[y]] <- sepset[[y]][[z]] <- S.j[,
                                                            i]
                Done <- TRUE
                break
              }
            }
          }
          indM <- which((apag[z, ] == 1 & apag[, z] ==
                           1) & (apag[y, ] == 1 & apag[, y] == 1))
          indM <- setdiff(indM, c(x, y, z))
          for (m in indM) {
            unshTripl <- cbind(unshTripl, c(y_, m, z_))
            unshVect <- c(unshVect, triple2numb(p, y_,
                                                m, z_))
            trueVstruct <- c(trueVstruct, TRUE)
          }
          indQ <- which((apag[z, ] == 1 & apag[, z] ==
                           1) & (apag[y, ] == 0 & apag[, y] == 0))
          indQ <- setdiff(indQ, c(x, y, z))
          for (q in indQ) {
            delTripl <- unshVect == (if (q < y)
              triple2numb(p, q, z, y)
              else triple2numb(p, y, z, q))
            if (any(delTripl))
              trueVstruct[which.max(delTripl)] <- FALSE
          }
          indR <- which((apag[z, ] == 0 & apag[, z] ==
                           0) & (apag[y, ] == 1 & apag[, y] == 1))
          indR <- setdiff(indR, c(x, y, z))
          for (r in indR) {
            delTripl <- unshVect == (if (r < z)
              triple2numb(p, r, y, z)
              else triple2numb(p, z, y, r))
            if (any(delTripl))
              trueVstruct[which.max(delTripl)] <- FALSE
          }
        }
        if (any(del1, del2))
          trueVstruct[k] <- FALSE
      }
    }
  }
  list(triple = unshTripl, vect = unshVect, sepset = sepset,
       apag = apag, trueVstruct = trueVstruct)
}
