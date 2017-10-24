# New model for HMC reuse
import pickle
from pystan import StanModel

model = """
data {
   int<lower=1> N; // Number of nodes
   matrix[N,N] W;  // the 1st predictor
   int<lower=1> B; // dimenstion of weights
}
parameters {
   matrix[N,2] l;
   real<lower=0> sigma;
   real<lower=0> eta;
   //real<lower=0> rho;
}
model {
   sigma ~ inv_gamma(1,1);
   for (i in 1:N)
        for (j in 1:2)
            l[i,j] ~ normal(0, sigma);
   for (i in 1:N)
        for (j in 1:N)
            W[i,j] ~ normal(0, eta*exp(-squared_distance(l[i,:],l[j,:])/10)); 
}
generated quantities {
    matrix[N,N] W_pred;
    for (i in 1:N)
        for (j in 1:N)
            W_pred[i,j] = normal_rng(0, exp(-squared_distance(l[i,:],l[j,:]))/10);
}
"""
sm = StanModel(model_code=model)

with open('model.pkl', 'wb') as f:
    pickle.dump(sm, f)



