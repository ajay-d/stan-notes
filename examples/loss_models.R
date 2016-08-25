#' http://www.magesblog.com/2015/10/non-linear-growth-curves-with-stan.html
#' http://www.magesblog.com/2015/11/loss-developments-via-growth-curves-and.html
#' http://www.magesblog.com/2015/11/hierarchical-loss-reserving-with-stan.html

library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

dat <- list(
  "N" = 27,  
  "x" =
    c(1, 1.5, 1.5, 1.5, 2.5, 4, 5, 5, 7, 8, 8.5, 9, 9.5, 9.5, 10, 
      12, 12, 13, 13, 14.5, 15.5, 15.5, 16.5, 17, 22.5, 29, 31.5),
  "Y" =
    c(1.8, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.47, 2.19, 
      2.26, 2.4, 2.39, 2.41, 2.5, 2.32, 2.32, 2.43, 2.47, 2.56, 2.65, 
      2.47, 2.64, 2.56, 2.7, 2.72, 2.57))

nlm <- nls(Y ~ alpha - beta * lambda^x, data=dat,
           start=list(alpha=1, beta=1, lambda=0.9))
summary(nlm)

stanmodel <- "
data {
  int<lower=0> N; 
  real x[N]; 
  real Y[N]; 
} 
parameters {
  real alpha; 
  real beta;  
  real<lower=.5,upper= 1> lambda; // orginal gamma in the JAGS example  
  real<lower=0> tau; 
} 
transformed parameters {
  real sigma; 
  sigma <- 1 / sqrt(tau); 
} 
model {
  real m[N];
  for (i in 1:N) 
    m[i] <- alpha - beta * pow(lambda, x[i]);
  
  Y ~ normal(m, sigma); 
  
  alpha ~ normal(0.0, 1000); 
  beta ~ normal(0.0, 1000); 
  lambda ~ uniform(.5, 1); 
  tau ~ gamma(.0001, .0001); 
}
generated quantities{
  real Y_mean[N]; 
  real Y_pred[N]; 
  for(i in 1:N){
    # Posterior parameter distribution of the mean
    Y_mean[i] <- alpha - beta * pow(lambda, x[i]);
    # Posterior predictive distribution
    Y_pred[i] <- normal_rng(Y_mean[i], sigma);   
}
}
"

fit <- stan(model_code = stanmodel, 
            model_name = "GrowthCurve", 
            data = dat)

Y_mean <- extract(fit, "Y_mean")
Y_mean_cred <- apply(Y_mean$Y_mean, 2, quantile, c(0.05, 0.95))
Y_mean_mean <- apply(Y_mean$Y_mean, 2, mean)

Y_pred <- extract(fit, "Y_pred")
Y_pred_cred <- apply(Y_pred$Y_pred, 2, quantile, c(0.05, 0.95))
Y_pred_mean <- apply(Y_pred$Y_pred, 2, mean)

alpha <- extract(fit, "alpha")$alpha
beta <- extract(fit, "beta")$beta
lambda <- extract(fit, "lambda")$lambda
tau <- extract(fit, "tau")$tau

str(Y_mean$Y_mean)
dat$x

y.1 <- Y_mean$Y_mean[,1]
yy.1 <- alpha - beta *lambda^(dat$x[[1]])

all.equal(y.1, as.numeric(yy.1), check.attributes = FALSE)

#' http://www.stat.columbia.edu/~gelman/research/published/stan_jebs_2.pdf

# Set up the true parameter values
a <- c(.8, 1)
b <- c(2, .1)
sigma <- .2
# Simulate data
x <- (1:1000)/100
N <- length(x)
ypred <- a[1]*exp(-b[1]*x) + a[2]*exp(-b[2]*x)
y <- ypred*exp(rnorm(N, 0, sigma))

stanmodel <- "
data {
int N;
vector[N] x;
vector[N] y;
}
parameters {
vector[2] log_a;
ordered[2] log_b;
real<lower=0> sigma;
}
transformed parameters {
vector<lower=0>[2] a;
vector<lower=0>[2] b;
a <- exp(log_a);
b <- exp(log_b);
}
model {
vector[N] ypred;
ypred <- a[1]*exp(-b[1]*x) + a[2]*exp(-b[2]*x);
y ~ lognormal(log(ypred), sigma);
}
"

# Fit the model
library("rstan")
fit <- stan(model_code=stanmodel, data=list(N=N, x=x, y=y), iter=1000, chains=4)
print(fit, pars=c("a", "b", "sigma"))

url <- "https://raw.githubusercontent.com/mages/diesunddas/master/Data/ClarkTriangle.csv"
dat <- read.csv(url)


####################################################################################################
library(rstan)
library(loo)

# Prepare data 
url <- "http://stat.columbia.edu/~gelman/arm/examples/arsenic/wells.dat"
wells <- read.table(url)
wells$dist100 <- wells$dist / 100 # rescale 
y <- wells$switch
a <- qlogis(mean(y)) # i.e., a = logit(Pr(y = 1))
x <- scale(model.matrix(~ 0 + dist + arsenic, wells))
data <- list(N = nrow(x), P = ncol(x), a = a, x = x, y = y)

logistic.stan <- "
data {
  int<lower=0> N; # number of data points
  int<lower=0> P; # number of predictors (including intercept)
  int<lower=0,upper=1> y[N]; # binary outcome
  matrix[N,P] x; # predictors (including intercept)
  real a;
}
parameters {
  real beta0;
  vector[P] beta;
}
model {
  beta0 ~ student_t(7, a, 0.1);
  beta ~ student_t(7, 0, 1);
  y ~ bernoulli_logit(beta0 + x * beta);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] <- bernoulli_logit_log(y[n], beta0 + x[n] * beta);
}
"
# Fit model
fit1 <- stan(model_code=logistic.stan, data = data)

# Extract log-likelihood and compute LOO
log_lik1 <- extract_log_lik(fit1)
loo1 <- loo(log_lik1) # or waic(log_lik1) to compute WAIC
waic1 <- waic(log_lik1)

# First run a second model using log(arsenic) instead of arsenic
data$x <- scale(model.matrix(~ 0 + dist100 + log(arsenic), wells))
fit2 <- stan(fit = fit1, data = data)
log_lik2 <- extract_log_lik(fit2)
loo2 <- loo(log_lik2)
waic2 <- waic(log_lik2)

# Compare
diff <- compare(loo1, loo2)