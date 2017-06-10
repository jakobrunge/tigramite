#' Generate random Fourier features
#' @param x Random variable x.
#' @param w Random coefficients.
#' @param b Random offsets.
#' @param num_f Number of random Fourier features.
#' @param sigma Smooth parameter of RBF kernel.
#' @param seed The seed for controlling random number generation. Use if you want to replicate results exactly. Default is NULL.
#' @return A list containing the p-value \code{p} and statistic \code{Sta}
#' @examples
#' random_fourier_features(rnorm(1000),num_f=25)

random_fourier_features <- function(x,w=NULL,b=NULL,num_f=NULL,sigma=NULL,seed=NULL){

  if (length(num_f)==0) num_f = 25;

  x = matrix2(x);

  r=nrow(x);
  c=ncol(x);

  if (sigma==0 | is.na(sigma)){sigma=1};

  if (length(w)==0){
    if (length(sigma)==0){sigma=1;}
    set.seed(seed)
    w=(1/sigma)*matrix(rnorm(num_f*c),num_f,c);
    set.seed(seed)
    b=repmat(2*pi*runif(num_f),1,r);
  } #else if (nrow(w)<num_f){
  #w=rbind(w,(1/sigma)*matrix(rnorm((num_f-nrow(w))*c),num_f-nrow(w),c));
  #b=rbind(b,repmat(2*pi*runif(num_f-nrow(b)),1,r));
  #}
  feat = sqrt(2)*t(cos(w[1:num_f,1:c]%*%t(x)+b[1:num_f,]));

  out=list(feat=feat,w=w,b=b);
  return(out)
}
