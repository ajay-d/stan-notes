data {
  int N_states;
  int N_industries;
  int N_all_years;
  int N_obs_years;
  int N_future_years;

  int N_obs;
  

  real obs_agg_loss_value[N_obs];
  int obs_agg_loss_state[N_obs];
  int obs_agg_loss_industry[N_obs];
  int obs_agg_loss_obs_year[N_obs];

  real log_premium[ N_obs_years, N_states, N_industries];

  // real log_diff_GSP[N_all_years, N_states, N_industries];
  real change_log_diff_GSP[N_all_years, N_states];
  //real log_diff_GSP[N_all_years];


  real v_factor[N_obs_years];
  
}

parameters {
  real log_mu0[N_states, N_industries];
  real log_mu0_mu;
  real<lower=0> log_mu0_sig;

  
  real long_trend_mu;
  real long_trend_stateFac[N_states];
  real long_trend_indFac[N_industries];
  real<lower=0> long_trend_stateFac_sig;
  real<lower=0> long_trend_indFac_sig;

  real<lower=-0.75> loading_on_state_trend[N_industries];
  vector[N_states] state_common_trend[N_obs_years-1];
  corr_matrix[N_states] Omega_state;
  real<lower=0, upper=1> state_common_trend_rho;
  real<lower=0> state_common_trend_sig;

  

  real<lower=-0.75> loading_on_industry_trend[N_states];
  vector[N_industries] industry_common_trend[N_obs_years-1];
  corr_matrix[N_industries] Omega_industry;
  real<lower=0, upper=1> industry_common_trend_rho;
  real<lower=0> industry_common_trend_sig;


  real change_log_diff_GSP_sensitivity;
 
  //real sensitivity_m0;

  real<lower=0> phi;
  real<lower=0.5, upper=2> p;

  
}


transformed parameters{
  real log_mu[N_obs_years, N_states, N_industries];
  real log_diff[N_obs_years, N_states, N_industries];
  real long_trend[N_states, N_industries];



  for(state_i in 1:N_states)
    for(ind_i in 1:N_industries) {
      long_trend[state_i,ind_i] <- long_trend_mu
                                 + long_trend_stateFac[state_i]
                                 + long_trend_indFac[ind_i];


    }

  
  for(state_i in 1:N_states)
    for(ind_i in 1:N_industries){
      int yr1;
      yr1 <- 1;
      
      log_diff[yr1, state_i, ind_i] <- 0;
      
      for(yr_i in 2:N_obs_years)
        {
          log_diff[yr_i, state_i, ind_i] <- long_trend[state_i, ind_i] 
                                          + loading_on_state_trend[ind_i] * state_common_trend[yr_i-1, state_i] 
                                          + loading_on_industry_trend[state_i] * industry_common_trend[yr_i-1, ind_i]

                                          + change_log_diff_GSP_sensitivity * change_log_diff_GSP[yr_i, state_i];
        }
    }
  

  {
    int yr1;
    yr1 <- 1;
    
    log_mu[yr1] <- log_mu0;
    
    for(yr_i in 2:N_obs_years)
      for(state_i in 1:N_states)
        for(ind_i in 1:N_industries)
          {           
            log_mu[yr_i,state_i, ind_i] 
              <-  log_mu[yr_i-1,state_i, ind_i] +
               log_diff[yr_i, state_i, ind_i];
          }
  }
  
  
}



model {
  matrix[N_industries, N_industries] SIG_industry;
  matrix[N_states, N_states] SIG_state;

  
  
  SIG_industry <- quad_form_diag(Omega_industry, rep_vector(industry_common_trend_sig, N_industries));
  SIG_state <- quad_form_diag(Omega_state, rep_vector(state_common_trend_sig, N_states));
  

  Omega_industry ~ lkj_corr(4);
  Omega_state ~ lkj_corr(4);

  
  p ~ normal(1, 1);
  phi ~ normal(1, 5);


  to_array_1d(log_mu0) ~ normal(log_mu0_mu, log_mu0_sig);
  log_mu0_sig ~ cauchy(0, 2); //bounded below by zero in parameter section
  log_mu0_mu ~ normal(0, 4); 


  
  long_trend_stateFac_sig ~ cauchy(0,2);
  long_trend_indFac_sig ~ cauchy(0,2);

  long_trend_mu ~ normal(0, 0.5);
  long_trend_stateFac ~ normal(0, long_trend_stateFac_sig);
  long_trend_indFac ~ normal(0, long_trend_indFac_sig);
  
  to_array_1d(loading_on_state_trend) ~ normal(1, 1);
  to_array_1d(loading_on_industry_trend) ~ normal(1, 1);
  
  state_common_trend_rho ~ beta(2, 1);//bounded: parameter section
  state_common_trend_sig ~ cauchy(0, 0.5); 
  industry_common_trend_rho ~ beta(2, 1);//bounded:  parameter section
  industry_common_trend_sig ~ cauchy(0, 0.5); 
  
  {
    int yr1;
    yr1 <- 1;
    state_common_trend[yr1] ~ multi_normal(rep_vector(0, N_states),
                                           quad_form_diag(Omega_state, 
                                                          rep_vector(state_common_trend_sig / (1-state_common_trend_rho^2), 
                                                                     N_states)));
    industry_common_trend[yr1] ~ multi_normal(rep_vector(0, N_industries),
                                              quad_form_diag(Omega_industry, 
                                                             rep_vector(industry_common_trend_sig / (1-industry_common_trend_rho^2), 
                                                                        N_industries)));

    for(yr_i in 2:(N_obs_years-1)){
      state_common_trend[yr_i] ~ multi_normal(state_common_trend_rho*state_common_trend[yr_i-1], SIG_state);
      industry_common_trend[yr_i] ~ multi_normal(industry_common_trend_rho*industry_common_trend[yr_i-1], SIG_industry);
      
    }
  }
  
  //log_diff_GSP_sensitivity ~ normal(0, 1);
  change_log_diff_GSP_sensitivity ~ normal(0, 4);
  //log_diff_GSP_sensitivity ~ normal(0, 4);
  //sensitivity_m0 ~ normal(0, 2);
  
  {
    vector[ N_obs] alpha_;
    vector[N_obs] beta_;
    vector[ N_obs] v_;
    vector[ N_obs] mu_scaled_;
    

    for(obs_i in 1:N_obs){
      int yr_i;
      int state_i;
      int ind_i;

      yr_i <- obs_agg_loss_obs_year[obs_i];
      state_i <- obs_agg_loss_state[obs_i];
      ind_i <- obs_agg_loss_industry[obs_i];
      
      mu_scaled_[obs_i] 
             <- exp(log_mu[yr_i, state_i, ind_i] 
                    + log_premium[yr_i, state_i, ind_i]);
      
      v_[obs_i] 
              <- v_factor[yr_i] * phi * exp(p * (log_mu[yr_i, state_i, ind_i] 
                                                 + log_premium[yr_i, state_i, ind_i]));
      
    }
    
    beta_ <- mu_scaled_ ./ v_;
    alpha_ <-mu_scaled_ .* beta_;
    
    
    obs_agg_loss_value ~ gamma(alpha_, beta_);
  }
}

generated quantities {
  vector[N_states] future_state_common_trend[ N_future_years];
  vector[N_industries] future_industry_common_trend[ N_future_years];

  real future_log_mu[ N_future_years, N_states, N_industries];
  real future_log_diff[ N_future_years, N_states, N_industries];


  
  {
    matrix[N_industries, N_industries] SIG_industry;
    matrix[N_states, N_states] SIG_state;
    int future_yr1;

    future_yr1 <- 1;    
    SIG_industry <- quad_form_diag(Omega_industry, rep_vector(industry_common_trend_sig, N_industries));
    SIG_state <- quad_form_diag(Omega_state, rep_vector(state_common_trend_sig, N_states));
  

    future_state_common_trend[future_yr1] <- multi_normal_rng(state_common_trend_rho*state_common_trend[N_obs_years-1], SIG_state);
    future_industry_common_trend[future_yr1] <- multi_normal_rng(industry_common_trend_rho*industry_common_trend[N_obs_years-1], SIG_industry);
    
    for(future_yr_i in 2:N_future_years){
      future_state_common_trend[future_yr_i] <- multi_normal_rng(state_common_trend_rho*future_state_common_trend[future_yr_i-1], SIG_state);
      future_industry_common_trend[future_yr_i] <- multi_normal_rng(industry_common_trend_rho*future_industry_common_trend[future_yr_i-1], SIG_industry);
      
    }
  }
 
  
  for(state_i in 1:N_states)
    for(ind_i in 1:N_industries){
      
      for(future_yr_i in 1:N_future_years)
        {
          future_log_diff[future_yr_i, state_i, ind_i] <- long_trend[state_i, ind_i] 
                                                        + loading_on_state_trend[ind_i] * future_state_common_trend[future_yr_i, state_i] 
                                                        + loading_on_industry_trend[state_i] * future_industry_common_trend[future_yr_i, ind_i]

                                                        + change_log_diff_GSP_sensitivity * change_log_diff_GSP[future_yr_i + N_obs_years, state_i];
        }
    }
  
 
  
  for(future_yr_i in 1:N_future_years)
    for(state_i in 1:N_states)
      for(ind_i in 1:N_industries)
        {
          
          if(future_yr_i ==1) {
            future_log_mu[future_yr_i,state_i, ind_i] 
              <-  log_mu[N_obs_years,state_i, ind_i] +
               future_log_diff[future_yr_i, state_i, ind_i];


          } else {
            future_log_mu[future_yr_i,state_i, ind_i] 
              <-  future_log_mu[future_yr_i-1,state_i, ind_i] +
               future_log_diff[future_yr_i, state_i, ind_i];
            
          }
        }
  

}
