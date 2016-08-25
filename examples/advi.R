library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())


stanmodel <- "
data {
  int<lower=0> J; // number of schools 
  real y[J]; // estimated treatment effects
  real<lower=0> sigma[J]; // s.e. of effect estimates 
}
parameters {
  real mu; 
  real<lower=0> tau;
  real eta[J];
}
transformed parameters {
  real theta[J];
  for (j in 1:J)
    theta[j] = mu + tau * eta[j];
}
model {
  target += normal_lpdf(eta | 0, 1);
  target += normal_lpdf(y | theta, sigma);
}"

schools_dat <- list(J = 8, 
                    y = c(28,  8, -3,  7, -1,  1, 18, 12),
                    sigma = c(15, 10, 16, 11,  9, 11, 10, 18))

fit <- stan(model_code = stanmodel, 
            data = schools_dat, 
            iter = 1000, chains = 4)

fit

school_model <- stan_model(model_code = stanmodel)
fit1 <- sampling(school_model, 
                 data = schools_dat, 
                 iter = 1000, chains = 4)
fit1
fit_advi <- vb(school_model, data = schools_dat)

