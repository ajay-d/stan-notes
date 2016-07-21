data {
  
  int<lower=1> N_series;
  int<lower=1> N_obs_years;
  int<lower=1> N_future_years;
  int<lower=1> N_states;
  int<lower=1> N_obs;
  
  vector[N_series] obs_growth[N_obs];
  real<lower=0> obs_employment[N_states, N_obs_years + N_future_years];
  
  int<lower=1> obs_state_map[N_obs];
  int<lower=1> obs_year_map[N_obs];

  real emp_growth[N_states, N_obs_years + N_future_years];
  real pct_const[N_states, N_obs_years + N_future_years];  
}

parameters {

  //serial correlation trend
  real<lower=-1,upper=1> rho1; 
  
  //common trends
  //one trend for all states
  vector[N_series] macro_hidden_uc_st1[N_obs_years];
  real<lower=0> macro_hidden_sig_st1;
  
  //vector[N_series] macro_hidden_uc_st2[N_obs_years];
  //real<lower=0,upper=1> macro_hidden_sig_st2;

  //loadings on common trends
  vector[N_states] st_loading_on_macro_process_st1[N_series];
  //vector [N_states-1] st_loading_on_macro_process_st2[N_series];

  //likelihood
  real sig_like_alpha;
  real sig_like_beta;
  
  //intercept
  vector[N_states] st_int[N_series, N_obs_years];
  //innovation
  vector<lower=0>[N_states] innov_sig;
  //real<lower=0> innov_sig_alpha;
  real<lower=0> innov_sig_beta;
  
  //coefficient on structural driver
  vector[N_states] beta_emp_growth_st[N_series];
  
  //all state coefficients shrunk to common estimate
  real beta_emp_growth_mu0[N_series];
  real<lower=0> beta_emp_growth_sig0[N_series];
  
   //coefficient on structural driver
  vector[N_states] beta_pct_const_st[N_series];
  
  //all state coefficients shrunk to common estimate
  real beta_pct_const_mu0[N_series];
  real<lower=0> beta_pct_const_sig0[N_series];
  

}

transformed parameters{
  
  vector [N_series] macro_hidden_st1[N_obs_years];
  //vector [N_series] macro_hidden_st2[N_obs_years];
  
  vector [N_states] st_loading_on_macro_process_st1_full[N_series];
  //vector [N_states] st_loading_on_macro_process_st2_full[N_series];
  
  //likelihood
  vector[N_series] sig_like[N_states, N_obs_years]; 
 
  //center the macro process
  for(ser_i in 1:N_series){
    real center_1;
	
    center_1 <- 0;
	
    for(yr_i in 1:N_obs_years) {
      center_1 <- center_1 + macro_hidden_uc_st1[yr_i][ser_i];
	  //center_2 <- center_2 + macro_hidden_uc_st2[yr_i][ser_i];
    }
	
    for(yr_i in 1:N_obs_years){
      macro_hidden_st1[yr_i][ser_i] <-  macro_hidden_uc_st1[yr_i][ser_i] - center_1 / N_obs_years;
	  //macro_hidden_st2[yr_i][ser_i] <-  macro_hidden_uc_st2[yr_i][ser_i] - center_2 / N_obs_years;
	}
	
	
    //For each Series
    for(yr_i in 1:N_obs_years)
      for(st_i in 1:N_states)
	    sig_like[st_i, yr_i][ser_i] <- exp(sig_like_alpha + log(obs_employment[st_i, yr_i])*sig_like_beta);
  }

}

model{

  //SD of AR(1) process
  for(ser_i in 1:N_series) {
    macro_hidden_uc_st1[1][ser_i] ~ normal(0, macro_hidden_sig_st1/sqrt(1-rho1*rho1));
    //for each series
    for(yr_i in 2:N_obs_years){
      macro_hidden_uc_st1[yr_i][ser_i] ~ normal(rho1 * macro_hidden_uc_st1[yr_i-1][ser_i], macro_hidden_sig_st1);
    }
  }
  
  
  //Enforce stationary AR(1) process
  rho1 ~ normal(0, 1)T[-1,1];
  
  //hidden
  macro_hidden_sig_st1 ~ cauchy(0,2)T[0,];
  //macro_hidden_sig_st2 ~ uniform(0, 1);  
  
  //loading
  //macro_loading_sig ~ beta(1.2, 2);

  
  //intercept for each state
  for(st_i in 1:N_states)
    innov_sig[st_i] ~ cauchy(.001,innov_sig_beta)T[0,];
    //innov_sig[st_i] ~ normal(innov_sig_alpha,innov_sig_beta)T[0,];

  innov_sig_beta ~ cauchy(.001,1)T[0,];
  //innov_sig_alpha ~ normal(.5,1)T[0,];
  //innov_sig_beta ~ normal(.5,1)T[0,];
  
  for(ser_i in 1:N_series){

    //loading
    for(st_i in 1:N_states){
		st_loading_on_macro_process_st1[ser_i][st_i] ~ normal(1, 1);
	}
	
	//intercept for each series / state
    for(st_i in 1:N_states){
	  st_int[ser_i, 1][st_i] ~ normal(0,5);
	  for(yr_i in 2:N_obs_years)
		st_int[ser_i, yr_i][st_i] ~ normal(st_int[ser_i, yr_i-1][st_i], innov_sig[st_i]);
	}
	
    //beta
    for(st_i in 1:N_states)
      beta_emp_growth_st[ser_i][st_i] ~ normal(beta_emp_growth_mu0[ser_i], beta_emp_growth_sig0[ser_i]);
      
    beta_emp_growth_mu0[ser_i] ~ normal(0,10);
    beta_emp_growth_sig0[ser_i] ~ cauchy(0,2)T[0,];
	
    for(st_i in 1:N_states)
      beta_pct_const_st[ser_i][st_i] ~ normal(beta_pct_const_mu0[ser_i], beta_pct_const_sig0[ser_i]);
      
    beta_pct_const_mu0[ser_i] ~ normal(0,10);
    beta_pct_const_sig0[ser_i] ~ cauchy(0,2)T[0,];

	
  }
  //likelihood for each series
  sig_like_alpha ~ normal(0,5);
  sig_like_beta ~ normal(0,5);
  
  
  
	
  //likelihood
  for(obs_i in 1:N_obs){
      int st_i;
      int yr_i;
	  vector[N_series] mu_i;
	  
      st_i <- obs_state_map[obs_i];
      yr_i <- obs_year_map[obs_i];
	  
      for(ser_i in 1:N_series) {
        mu_i[ser_i] <- st_int[ser_i, yr_i][st_i] + 
				             
                       emp_growth[st_i, yr_i] * beta_emp_growth_st[ser_i][st_i] +
					   pct_const[st_i, yr_i] * beta_pct_const_st[ser_i][st_i] +
					 
                       macro_hidden_st1[yr_i][ser_i] * st_loading_on_macro_process_st1[ser_i][st_i]
					   
					   ;
	
		//obs_growth[obs_i][ser_i] ~ normal(mu_i[ser_i], sig_like[st_i][ser_i]);
	  }
	  
	  //for each series
	  obs_growth[obs_i] ~ normal(mu_i, sig_like[st_i, yr_i]);
  }
  
}

generated quantities{

  vector[N_series] macro_hidden_full_st1[N_obs_years + N_future_years];
  //vector[N_series] macro_hidden_full_st2[N_obs_years + N_future_years];
  vector[N_states] st_int_full[N_series, N_obs_years + N_future_years];
  
  for(ser_i in 1:N_series)
    for(yr_i in 1:N_obs_years){
      macro_hidden_full_st1[yr_i][ser_i] <- macro_hidden_st1[yr_i][ser_i];
	  //macro_hidden_full_st2[yr_i][ser_i] <- macro_hidden_st2[yr_i][ser_i];
	  st_int_full[ser_i, yr_i] <- st_int[ser_i, yr_i];
      
    }

  
  
  for(ser_i in 1:N_series)
    for(yr_i in (N_obs_years+1):(N_obs_years+N_future_years)) {
      macro_hidden_full_st1[yr_i][ser_i] <- normal_rng(rho1 * macro_hidden_full_st1[yr_i-1][ser_i],macro_hidden_sig_st1);
	  //macro_hidden_full_st2[yr_i][ser_i] <- normal_rng(rho2[ser_i] .* macro_hidden_full_st2[yr_i-1][ser_i],macro_hidden_sig_st2);
	  for(st_i in 1:N_states)
		st_int_full[ser_i, yr_i][st_i] <- normal_rng(st_int_full[ser_i, yr_i-1][st_i], innov_sig[st_i]);
	  
	}
	
}



