
Sta_perm <- function(f_x,f_y,r){

Cxy=cov(f_x,f_y);

Sta = r*sum(Cxy^2);

return(Sta)

}
