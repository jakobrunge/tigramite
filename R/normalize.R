#' Normalizes each column of a matrix to mean zero unit variance.
#' @param mat Matrix
#' @return A matrix where each columns has mean zero and unit variance.
#' @examples
#' normalize(matrix(rnorm(10,2),ncol=2))


normalize <- function(mat){

  if (is.null(nrow(mat))){mat = matrix(mat);}

  mat = apply(mat, 2, function(x) if (sd(x)>0){(x - mean(x)) / sd(x)} else{x-mean(x);})


}
