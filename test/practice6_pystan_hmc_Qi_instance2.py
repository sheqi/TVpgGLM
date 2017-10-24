# New model for HMC reuse
import pickle
from pystan import StanModel

model = """
data {
   int<lower=1> N; // Number of data points
   matrix[N,N] W;  // the 1st predictor
   int A[N,N];     // the 2nd predictor
}
parameters {
   matrix[N,2] l;
   real<lower=0> sigma;
   real<lower=0> p;
   real<lower=0> eta;
}
model {
   sigma ~ inv_gamma(1,1);
   p ~ beta(1,1);
   for (i in 1:N)
        for (j in 1:2)
            l[i,j] ~ normal(0, sigma);
   for (i in 1:N)
        for (j in 1:N)
            A[i,j] ~ bernoulli(p);
   for (i in 1:N)
        for (j in 1:N)
            if (A[i,j] == 1)
                W[i,j] ~ normal(0, eta*exp(-squared_distance(l[i,:],l[j,:])));           
}
"""

sm = StanModel(model_code=model)

with open('model2.pkl', 'wb') as f:
    pickle.dump(sm, f)