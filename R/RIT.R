#' Tests whether x and y are unconditionally independent
#' @param x Random variable x.
#' @param y Random variable y.
#' @param approx Method for approximating the null distribution. Options include:
#' "lpd4," the Lindsay-Pilla-Basak method (default),
#' "gamma" for the Satterthwaite-Welch method,
#' "hbe" for the Hall-Buckley-Eagleson method,
#' "chi2" for a normalized chi-squared statistic,
#' "perm" for permutation testing (warning: this one is slow but recommended for small samples generally <500 )
#' @param seed The seed for controlling random number generation. Use if you want to replicate results exactly. Default is NULL.
#' @return A list containing the p-value \code{p} and statistic \code{Sta}
#' @examples
#' RIT(rnorm(1000),rnorm(1000));
#'
#' x=rnorm(1000);
#' y=(x+rnorm(1000))^2;
#' RIT(x,y);

RIT <- function(x,y,approx="lpd4",seed=NULL){

  if (sd(x)==0 | sd(y)==0){
    out=list(p=1,Sta=0,w=w,b=b);
    return(out$p)
  }

  x=matrix2(x);
  y=matrix2(y);


  r=nrow(x);
  if (r>500){
    r1=500
  } else {r1=r;}

  x=normalize(x);
  y=normalize(y);

  four_x = random_fourier_features(x,num_f=5,sigma=median(c(t(dist(x[1:r1,])))), seed = seed );
  four_y = random_fourier_features(y,num_f=5,sigma=median(c(t(dist(y[1:r1,])))), seed = seed );

  f_x=normalize(four_x$feat); #stabilizes computations
  f_y=normalize(four_y$feat); #stabilizes computations

  Cxy=cov(f_x,f_y);

  Sta = r*sum(Cxy^2);

  #approximate null distributions

  if (approx == "perm"){
    nperm =100;

    Stas = c();
    for (ps in 1:nperm){
      perm = sample(1:r,r);
      Sta_p = Sta_perm(f_x[perm,],f_y,r)
      Stas = c(Stas, Sta_p);

    }

    p = 1-(sum(Sta >= Stas)/length(Stas));

  } else{

    res_x = f_x-repmat(t(matrix(colMeans(f_x))),r,1);
    res_y = f_y-repmat(t(matrix(colMeans(f_y))),r,1);

    d =expand.grid(1:ncol(f_x),1:ncol(f_y));
    res = res_x[,d[,1]]*res_y[,d[,2]];
    Cov = 1/r * (t(res)%*%res);

    if (approx == "chi2"){
    i_Cov = ginv(Cov)

    Sta = r * (c(Cxy)%*%  i_Cov %*% c(Cxy) );
    p = 1-pchisq(Sta, length(c(Cxy)));
  } else{

    eig_d = eigen(Cov);
    eig_d$values=eig_d$values[eig_d$values>0];

    if (approx == "gamma"){
      p=1-sw(eig_d$values,Sta);

    } else if (approx == "hbe") {

      p=1-hbe(eig_d$values,Sta);

    } else if (approx == "lpd4"){
      eig_d_values=eig_d$values;
      p=try(1-lpb4(eig_d_values,Sta), silent=TRUE);
      if (!is.numeric(p)){
        p=1-hbe(eig_d$values,Sta);
      }
    }
  }
  }

  if (p<0) p=0;

  out=list(p=p,Sta=Sta);
  return(out)
}
