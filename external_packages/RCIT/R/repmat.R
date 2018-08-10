#' Repeat rows or columns of a vector or matrix
#' @param X Vector or matrix
#' @param m number of row copies
#' @param n number of column copies
#' @return Matrix.
#' @examples
#' repmat(matrix(rnorm(10*2),ncol=2),2,2)

repmat = function(X,m,n){
  ##R equivalent of repmat (matlab)
  if (!is.matrix(X)){X=matrix(X);}
  mx = dim(X)[1]
  nx = dim(X)[2]
  matrix(t(matrix(X,mx,nx*n)),mx*m,nx*n,byrow=T)
}
