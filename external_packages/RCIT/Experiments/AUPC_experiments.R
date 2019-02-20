### AUPC EXPERIMENTS ###

### sample size

samps = c(500,1000,2000,5000,1E4,1E5,1E6); # all sample sizes

# matrices to store results
res_RCIT_d = matrix(0,length(samps),1000);
res_RCoT_d = matrix(0,length(samps),1000);
res_RCIT_t_d = matrix(0,length(samps),1000);
res_RCoT_t_d = matrix(0,length(samps),1000);

res_KCIT_d = matrix(0,length(samps),1000);
res_KCIT_t_d = matrix(0,length(samps),1000);

res_RCIT_d_perm = matrix(0,length(samps),1000);
res_RCoT_d_perm = matrix(0,length(samps),1000);
res_RCIT_t_d_perm = matrix(0,length(samps),1000);
res_RCoT_t_d_perm = matrix(0,length(samps),1000);

for (n in 1:length(samps)){ # for each sample size
  print(n)
  for (t in 1:1000){ # repeat 100 times

    outer_x = sample(c(1:5),1); # sample outer non-linear function
    outer_y = sample(c(1:5),1);

    ex=rnorm(samps[n]); # sample errors for X
    ey=rnorm(samps[n]); # sample errors for Y
    eb=0.25*rnorm(samps[n]); # small dependence inducing error terms

    z=rnorm(samps[n]); #instantiate Z

    x = ex+eb;  #instantiate X
    y = ey+eb;  #instantiate Y

    #pass X through randomly chosen non-linear function
    if (outer_x == 2){
      x = (x)^2;
    } else if (outer_x == 3){
      x = (x)^3;
    } else if (outer_x == 4){
      x = exp(-sqrt(x^2));
    } else if (outer_x == 5){
      x = tanh(x);
    }

    #pass Y through randomly chosen non-linear function
    if (outer_y == 2){
      y = (y)^2;
    } else if (outer_y == 3){
      y = (y)^3;
    } else if (outer_y == 4){
      y = exp(-sqrt(y^2));
    } else if (outer_y == 5){
      y = tanh(y);
    }

    #run RCIT
    start=proc.time();
    out=RCIT(x,y,z);
    end=proc.time()-start;
    res_RCIT_d[n,t]=out$p; #store p-value
    res_RCIT_t_d[n,t]=end[3]; #store timing result

    #run RCoT
    start=proc.time();
    out_g = RCoT(x,y,z);
    end=proc.time()-start;
    res_RCoT_d[n,t]=out_g$p;
    res_RCoT_t_d[n,t]=end[3];

    if (samps[n]<=1E4){
      #run permutation test with S
      start=proc.time();
      out=RCIT(x,y,z,approx="perm");
      end=proc.time()-start;
      res_RCIT_d_perm[n,t]=out$p;
      res_RCIT_t_d_perm[n,t]=end[3];

      #run permutation test with S'
      start=proc.time();
      out_g = RCoT(x,y,z,approx="perm");
      end=proc.time()-start;
      res_RCoT_d_perm[n,t]=out_g$p;
      res_RCoT_t_d_perm[n,t]=end[3];
    }

    #run KCIT
    if (samps[n]<=2000){
      start=proc.time();
      res_KCIT_d[n,t]=KCIT(x,y,z);
      end=proc.time()-start;
      res_KCIT_t_d[n,t]=end[3];
    }

  }

  save(res_RCIT_d, res_RCoT_d, res_KCIT_d,
       res_RCIT_t_d, res_RCoT_t_d, res_KCIT_t_d,
       res_RCIT_d_perm, res_RCoT_d_perm, res_RCIT_t_d_perm, res_RCoT_t_d_perm,
       file="Type2_SS.RData")

}


### conditioning set size

dims = 1:10; #1-10 dimensions in conditioning set
samps=1000; #number of samples

## matrices to store results
res_RCIT_d = matrix(0,length(dims),1000);
res_RCoT_d = matrix(0,length(dims),1000);
res_RCIT_t_d = matrix(0,length(dims),1000);
res_RCoT_t_d = matrix(0,length(dims),1000);

res_KCIT_d = matrix(0,length(dims),1000);
res_KCIT_t_d = matrix(0,length(dims),1000);

res_RCIT_d_perm = matrix(0,length(dims),1000);
res_RCoT_d_perm = matrix(0,length(dims),1000);
res_RCIT_t_d_perm = matrix(0,length(dims),1000);
res_RCoT_t_d_perm = matrix(0,length(dims),1000);


for (n in dims){ # for each dimension
  print(n)
  for (t in 1:1000){ # repeat 1000 times

    outer_x = sample(c(1:5),1); # sample outer non-linear function
    outer_y = sample(c(1:5),1);

    ex=rnorm(samps); #sample errors for X
    ey=rnorm(samps); #sample errors for Y
    eb=0.25*rnorm(samps); #small dependence inducing error terms

    z=matrix(rnorm(samps*(dims[n])),ncol=dims[n]); #instantiate Z

    x = ex+eb;  #instantiate X
    y = ey+eb;  #instantiate Y

    #pass X through randomly chosen non-linear function
    if (outer_x == 2){
      x = (x)^2;
    } else if (outer_x == 3){
      x = (x)^3;
    } else if (outer_x == 4){
      x = exp(-sqrt(x^2));
    } else if (outer_x == 5){
      x = tanh(x);
    }

    #pass Y through randomly chosen non-linear function
    if (outer_y == 2){
      y = (y)^2;
    } else if (outer_y == 3){
      y = (y)^3;
    } else if (outer_y == 4){
      y = exp(-sqrt(y^2));
    } else if (outer_y == 5){
      y = tanh(y);
    }

    #run RCIT
    start=proc.time();
    out=RCIT(x,y,z);
    end=proc.time()-start;
    res_RCIT_d[n,t]=out$p; #store p-value
    res_RCIT_t_d[n,t]=end[3]; #store timing result

    #run RCoT
    start=proc.time();
    out_g = RCoT(x,y,z);
    end=proc.time()-start;
    res_RCoT_d[n,t]=out_g$p;
    res_RCoT_t_d[n,t]=end[3];

    #run permutation test with S
    start=proc.time();
    out=RCIT(x,y,z,approx="perm");
    end=proc.time()-start;
    res_RCIT_d_perm[n,t]=out$p;
    res_RCIT_t_d_perm[n,t]=end[3];

    #run permutation test with S'
    start=proc.time();
    out_g = RCoT(x,y,z,approx="perm");
    end=proc.time()-start;
    res_RCoT_d_perm[n,t]=out_g$p;
    res_RCoT_t_d_perm[n,t]=end[3];

    #run KCIT
    start=proc.time();
    res_KCIT_d[n,t]=KCIT(x,y,z);
    end=proc.time()-start;
    res_KCIT_t_d[n,t]=end[3];


  }

  save(res_RCIT_d, res_RCoT_d, res_KCIT_d,
       res_RCIT_t_d, res_RCoT_t_d, res_KCIT_t_d,
       res_RCIT_d_perm, res_RCoT_d_perm, res_RCIT_t_d_perm, res_RCoT_t_d_perm,
       file="Type2_Dim.RData")

}
