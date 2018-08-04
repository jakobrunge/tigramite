RBF_kernel <- function(x,width){

  n2=(dist(cbind(x,x),diag = TRUE, upper = TRUE))^2;
  n2=as.matrix(n2);

  width=width/2;
  kx = exp(-n2*width);

  return(kx)

}
