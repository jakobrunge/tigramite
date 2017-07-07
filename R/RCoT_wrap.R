RCoT_wrap<-function(x,y,z,suffStat){

 x=suffStat$data[,x];
 y=suffStat$data[,y];
 z=suffStat$data[,z];

 out = RCIT:::RCoT(x,y,z,seed=1);

 return(out$p)

}

