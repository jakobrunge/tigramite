#' KCIT - tests whether x and y are conditionally independent given z. Calls U_KCI if z is empty.
#' @param x Random variable x.
#' @param y Random variable y.
#' @param z Random variable z.
#' @param Bootstrap Estimate the null by bootstrap. Default is TRUE. Runs slightly faster when set to FALSE but with less accurate p-values.
#' @return The p-value \code{p}
#' @export

KCIT <- function(x,y,z,Bootstrap=TRUE){

  if( length(z)==0 || missing(z) ){
    p_val = U_KCI(x,y);
    #print(p_val)
    return(p_val)
  }
  else{

    x=normalize(x);
    y=normalize(y);
    z=normalize(z);

    T=length(x);

    lambda = 1E-3;
    Thresh = 1E-5;

    if (T <= 200) {
      #width = 0.8;
      width = 1.2;
    } else if (T > 1200){
      #width = 0.3;
      width = 0.7;
    }else{
      #width = 0.5;
      width = 0.4;
      }

    if (length(x)>500){
    width = median(as.vector(dist(cbind(x[1:500],y[1:500]))));}
    else {
    width = median(as.vector(dist(cbind(x,y))));}

    if (is.null(dim(z)[2])) { D = 1;}
    else{D = dim(z)[2];}

    theta = 1/(width^2 * D);

    H =  diag(T) - (1/T)*matrix(1,T,T);

    Kx = RBF_kernel(cbind(x, z/2), theta); Kx = H %*% Kx %*% H;

    Ky = RBF_kernel(y, theta); Ky = H %*% Ky %*% H;

    Kz = RBF_kernel(z, theta); Kz = H %*% Kz %*% H;

    #P1 = (diag(T)-Kz %*% chol2inv(chol(Kz + lambda*diag(T))));
    P1 = (diag(T)-Kz %*% solve(Kz + lambda*diag(T)));
    Kxz = P1 %*% Kx %*% t(P1);
    Kyz = P1 %*% Ky %*% t(P1);

    Sta = sum(diag(Kxz %*% Kyz));

    df = sum(diag(diag(T)-P1));

    listxz = eigen((Kxz+t(Kxz))/2,symmetric=TRUE);
    eig_Kxz=listxz[[1]]; eivx=listxz[[2]]

    listyz = eigen((Kyz+t(Kyz))/2,symmetric=TRUE);
    eig_Kyz=listyz[[1]]; eivy=listyz[[2]]


    IIx = which(eig_Kxz > max(eig_Kxz) * Thresh);
    IIy = which(eig_Kyz > max(eig_Kyz) * Thresh);
    eig_Kxz = eig_Kxz[IIx];
    eivx = eivx[,IIx];
    eig_Kyz = eig_Kyz[IIy];
    eivy = eivy[,IIy];

    eiv_prodx = eivx %*% diag(sqrt(eig_Kxz));
    eiv_prody = eivy %*% diag(sqrt(eig_Kyz));

    Num_eigx = dim(eiv_prodx)[2];
    Num_eigy = dim(eiv_prody)[2];
    Size_u = Num_eigx * Num_eigy;

    uu = matrix(0,T,Size_u);


    for (i in 1:Num_eigx){
      for (j in 1:Num_eigy){
        uu[,(i-1)*Num_eigy + j] = eiv_prodx[,i] * eiv_prody[,j];
      }
    }

    if (Size_u > T){
      uu_prod = uu %*% t(uu);}
    else{
      uu_prod = t(uu) %*% uu;
    }

    if (Bootstrap){
      T_BS=5000;
      IF_unbiased=TRUE;

      list_uu = eigen(uu_prod);
      eig_uu =list_uu[[1]];
      II_f = which(eig_uu > max(eig_uu) * Thresh);
      eig_uu = eig_uu[II_f];

      if (length(eig_uu)*T < 1E6){
        f_rand1 = matrix(rnorm(length(eig_uu)*T_BS)^2,length(eig_uu),T_BS);
        Null_dstr = t(eig_uu) %*% f_rand1;

      } else {

        Null_dstr = matrix(0,1,T_BS);
        Length = max(c(floor(1E6/T),100));
        Itmax = floor(length(eig_uu)/Length);
        for (iter in 1:Itmax){
          f_rand1 = matrix(rnorm(Length*T_BS)^2,Length,T_BS);
          Null_dstr = Null_dstr + t(eig_uu[((iter-1)*Length+1):(iter*Length)]) %*% f_rand1;

        }
      }

      sort_Null_dstr = sort(Null_dstr);
      #Cri = sort_Null_dstr[ceiling((1-alpha)*T_BS)];
      p_val = sum(Null_dstr>Sta)/T_BS;
      #print(p_val)
      return(p_val)

    } else {
      mean_appr = sum(diag(uu_prod));
      var_appr = 2*sum(diag(uu_prod^2));
      k_appr = mean_appr^2/var_appr;
      theta_appr = var_appr/mean_appr;
      p_val = 1-pgamma(Sta, shape=k_appr, scale=theta_appr);
      #print(p_val)
      return(p_val)

    }

  }

}


