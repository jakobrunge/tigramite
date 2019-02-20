#' Converts a vector to a matrix, or does nothing if the input is a matrix.
#' @param mat Vector or matrix.
#' @return Matrix.
#' @examples
#' matrix2(rnorm(10));
#' matrix2(matrix(rnorm(10,2),ncol=2));

matrix2 <- function(mat)

  if (is.matrix(mat)) {
    return(mat);
  } else {
    mat = matrix(mat);
    return(mat); }
