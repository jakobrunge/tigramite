#' Repeat copies of array like MATLAB's repmat function
#' @param X a matrix or a vector
#' @param m Number of row repeats
#' @param n Number of column repeats
#' @return A matrix with associated row and column repeats
#' @examples
#' repmat(matrix(rnorm(2*2),ncol=2),m=2,n=3)


repmat = function(X,m,n){
  ##R equivalent of repmat (matlab)
  X=matrix2(X);
  mx = dim(X)[1]
  nx = dim(X)[2]
  matrix(t(matrix(X,mx,nx*n)),mx*m,nx*n,byrow=T)
}
