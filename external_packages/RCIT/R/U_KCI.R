#' RCIT and RCoT - tests whether x and y are unconditionally independent..
#' @param x Random variable x.
#' @param y Random variable y.
#' @return The p-value \code{p}
#' @export

U_KCI <-function(x,y){

if (sd(x)>0) x=RCIT:::normalize(x);
if (sd(y)>0) y=RCIT:::normalize(y);


T=length(x);

if (length(x)>500){
  width = sqrt(2)*median(as.vector(dist(cbind(x[1:500],y[1:500]))));}
else {
  width = sqrt(2)*median(as.vector(dist(cbind(x,y))));}

theta = 1/(width^2);

H =  diag(T) - (1/T)*matrix(1,T,T);

Kx = RBF_kernel(x, theta); Kx = H %*% Kx %*% H;
Ky = RBF_kernel(y, theta); Ky = H %*% Ky %*% H;

Sta = sum(diag(Kx %*% Ky));

mean_appr = sum(diag(Kx)) * sum(diag(Ky)) /T;
var_appr = 2* sum(diag(Kx %*% Kx)) %*% sum(diag(Ky %*% Ky))/T^2;
k_appr = (mean_appr^2)/var_appr;
theta_appr = var_appr/mean_appr;


p_val = 1-pgamma(Sta, shape=k_appr, scale=theta_appr);

return(p_val)

}
