/*
 *The model is a loss ratio trend model (specifically for WC).
 *The model fits to log growth rates for medical severity, indemnity severity, and frequency by state and industry.
 *The model is a Dynamic Factor model.
 *The is a common trend at the macro level, state level (one for each state), and industry level (one for each industry). 

 *st -> state
 *ind -> industry
 *yr -> year
 */

data {
  
  int<lower=1> N_series;
  int<lower=1> freq_series_id;
  int<lower=1> med_series_id;
  int<lower=1> idm_series_id;
  
  int<lower=1> N_obs_years;
  int<lower=1> N_future_years;
  int<lower=1> N_states;
  int<lower=1> N_industries;
  
  int<lower=1> N_obs;
  vector[N_series] obs_growth[N_obs];
  int<lower=1> obs_state_map[N_obs];
  int<lower=1> obs_industry_map[N_obs];
  int<lower=1> obs_year_map[N_obs];


  real change_in_emp_growth_ID_st_ind_yr[N_states, N_industries, N_obs_years + N_future_years]; 
  real emp_growth_ID_st_ind_yr[N_states, N_industries, N_obs_years + N_future_years]; 
}

parameters {

  vector<lower=0,upper=1>[N_series] rho; //serial corelation in the macro trend
  vector<lower=0,upper=1>[N_series] rho_ind; //serial correlation in the indstry level trends


  //common trends
  vector[N_series]  macro_hidden_uc[N_obs_years]; // all states load of of this trend
  vector[N_series]  st_hidden_uc[N_states, N_obs_years]; //common trend in each state
  vector[N_series]  ind_hidden_uc[N_industries, N_obs_years]; //common trend in each industry
  
  real<lower=0,upper=1> macro_hidden_innov_sig;
  real<lower=0,upper=1> st_hidden_innov_sig;
  real<lower=0,upper=1> ind_hidden_innov_sig;
  
  //loadings on common trends
  //all loadins are shrunk toward 1.0 -> gives the common trend a "scale"
  //same loading for macro regardless of industry
  vector<lower=0>[N_states] macro_loading[N_series]; // no counter cyclical

  //loading on the state common trend
  matrix<lower=0>[N_states,N_industries] st_ind_loading_on_st_process[N_series]; //no counter cyclical

  //loading on the industry common trend
  matrix<lower=0>[N_states,N_industries] st_ind_loading_on_ind_process[N_series]; //no counter cyclical
  
  real<lower=0,upper=1> macro_loading_sig;
  real<lower=0,upper=1> st_loading_sig;
  real<lower=0,upper=1> ind_loading_sig;
  
  //mesurement error sd
  matrix<lower=0>[N_states, N_industries] sig_ID_st_ind[N_series];
  real<lower=0> sig_hyper_sig[N_series];
  real<lower=0> sig_hyper_mu[N_series];


  //intercept process
  real mu0[N_series];
  vector[N_states] st_int_fact[N_series];
  vector[N_industries] ind_int_fact[N_series];
  matrix[N_states, N_industries] intercept_e_ID_st_ind[N_series];

  real<lower=0,upper=1> st_int_fact_sig[N_series];
  real<lower=0,upper=1> ind_int_fact_sig[N_series];

  //shrinkage for intercept
  real<lower=0,upper=1> intercept_e_sig[N_series];

  //we assume that the impact of employment growth is similar within an industry regardless of what state you are in
  //    (so we shrink to a common industry mean)
  matrix[N_states,N_industries] beta_emp_change_ID_st_ind[N_series];
  vector[N_industries] beta_emp_change_mu0[N_series];
  real<lower=0,upper=1> beta_emp_change_sig0[N_series];
  real beta_emp_change_mu00[N_series];
  real<lower=0,upper=1> beta_emp_change_sig00[N_series];

  matrix[N_states,N_industries] beta_emp_growth_ID_st_ind[N_series];
  vector[N_industries] beta_emp_growth_mu0[N_series];
  real<lower=0,upper=1> beta_emp_growth_sig0[N_series];
  real beta_emp_growth_mu00[N_series];
  real<lower=0,upper=1> beta_emp_growth_sig00[N_series];
  

  corr_matrix[N_series] macro_Omega;
  corr_matrix[N_series] st_Omega;
  corr_matrix[N_series] ind_Omega;
  corr_matrix[N_series] e_Omega;
 
}

transformed parameters{
  vector[N_series] macro_hidden[N_obs_years];
  vector[N_series] st_hidden[N_states, N_obs_years];
  vector[N_series] ind_hidden[N_industries, N_obs_years];

  matrix[N_states, N_industries] intercept_ID_st_ind[N_series];
  

  /*center the macro process*/
  for(ser_i in 1:N_series){
    real center;
    center <- 0;
    for(yr_i in 1:N_obs_years)
      center <- center +  macro_hidden_uc[yr_i][ser_i];
    
    for(yr_i in 1:N_obs_years)
      macro_hidden[yr_i][ser_i] <-  macro_hidden_uc[yr_i][ser_i] - center / N_obs_years;
  }

  /*center each state process*/
  for(ser_i in 1:N_series){
    for(st_i in 1:N_states){
      real center;
      center <- 0;

      for(yr_i in 1:N_obs_years)
        center <- center +  st_hidden_uc[st_i,yr_i][ser_i];
      
      for(yr_i in 1:N_obs_years)
        st_hidden[st_i,yr_i][ser_i] <-  st_hidden_uc[st_i,yr_i][ser_i] - center / N_obs_years;
    }
  }

  /*center each industry process*/
  for(ser_i in 1:N_series){
    for(ind_i in 1:N_industries){
      real center;
      center <- 0;
      
      for(yr_i in 1:N_obs_years)
        center <- center +  ind_hidden_uc[ind_i,yr_i][ser_i];
      
      for(yr_i in 1:N_obs_years)
        ind_hidden[ind_i,yr_i][ser_i] <-  ind_hidden_uc[ind_i,yr_i][ser_i] - center / N_obs_years;
    }
  }


  
  //intercept is a factor model with shrinkage 
  for(ser_i in 1:N_series)
    for(st_i in 1:N_states)
      for(ind_i in 1:N_industries)
        intercept_ID_st_ind[ser_i][st_i, ind_i] <- mu0[ser_i] +  
                                                 st_int_fact[ser_i][st_i]+
                                                 ind_int_fact[ser_i][ind_i] +
                                                 intercept_e_ID_st_ind[ser_i][st_i,ind_i];

  
  
}



model{
  
  macro_Omega ~ lkj_corr(1);
  ind_Omega ~ lkj_corr(1);
  st_Omega ~ lkj_corr(1);
  e_Omega ~ lkj_corr(1);


  //hidden processes (before centering)

  { // macro

    vector[N_series]  m_sig_0;
    //matrix[N_series, N_series] S;
    matrix[N_series, N_series] L;
    

    //S <- diag_matrix(macro_hidden_innov_sig) * 
    //   macro_Omega *
    //   diag_matrix(macro_hidden_innov_sig);
   
    L <- cholesky_decompose(macro_Omega);
    for (s1 in 1:N_series)
      for (s2 in 1:s1)
        L[s1,s2] <- macro_hidden_innov_sig * L[s1,s2];
    
    for(ser_i in 1:N_series)
      m_sig_0[ser_i] <- macro_hidden_innov_sig/sqrt(1-rho[ser_i]*rho[ser_i]);
    
    macro_hidden_uc[1] ~ multi_normal(rep_vector(0.0, N_series), 
                                      diag_matrix(m_sig_0)*
                                      macro_Omega *
                                      diag_matrix(m_sig_0));
    
    

    for(yr_i in 2:N_obs_years)
      macro_hidden_uc[yr_i] ~ multi_normal_cholesky(rho .* macro_hidden_uc[yr_i-1],L);
  }


  //hidden process for each state
  {
    //matrix[N_series, N_series] S;
    matrix[N_series, N_series] L;

    //S <- diag_matrix(st_hidden_innov_sig) * 
    //   st_Omega *
    //   diag_matrix(st_hidden_innov_sig);

    L <- cholesky_decompose(st_Omega);
    for (s1 in 1:N_series)
      for (s2 in 1:s1)
        L[s1,s2] <- st_hidden_innov_sig * L[s1,s2];
    
    for(st_i in 1:N_states){
      st_hidden_uc[st_i,1] ~ multi_normal(rep_vector(0.0,N_series),
                                          st_hidden_innov_sig * st_hidden_innov_sig * N_obs_years *
                                          st_Omega);
      
      for(yr_i in 2:N_obs_years)
        st_hidden_uc[st_i,yr_i] ~ multi_normal_cholesky(st_hidden_uc[st_i,yr_i-1],L);
    }  
  }

  //hidden process for each ind
  {
    vector[N_series]  i_sig_0;
    //matrix[N_series, N_series] S;
    matrix[N_series, N_series] L;

    //S <- diag_matrix(ind_hidden_innov_sig) * 
    //   ind_Omega *
    //   diag_matrix(ind_hidden_innov_sig);

    L <- cholesky_decompose(ind_Omega);
    for (s1 in 1:N_series)
      for (s2 in 1:s1)
        L[s1,s2] <- ind_hidden_innov_sig * L[s1,s2];

    for(ser_i in 1:N_series)
      i_sig_0[ser_i] <- ind_hidden_innov_sig/sqrt(1-rho_ind[ser_i]*rho_ind[ser_i]);

    for(ind_i in 1:N_industries){
      ind_hidden_uc[ind_i,1] ~ multi_normal(rep_vector(0.0,N_series),
                                           diag_matrix(i_sig_0) * 
                                           ind_Omega *
                                           diag_matrix(i_sig_0));
      
      for(yr_i in 2:N_obs_years)
        ind_hidden_uc[ind_i,yr_i] ~ multi_normal_cholesky(rho_ind .* ind_hidden_uc[ind_i,yr_i-1],L);
    }  
  }



  rho ~ beta(1.2, 1.2); //one for each series
  rho_ind ~ beta(3, 1);//one for each series
  
  
  /*sig*/
  
  //hidden
  macro_hidden_innov_sig ~ beta(1, 2); 
  st_hidden_innov_sig ~ beta(1, 2);
  ind_hidden_innov_sig ~ beta(1, 2);
  
  //loading
  macro_loading_sig ~ beta(1.2, 2);
  st_loading_sig ~ beta(1.2, 2);
  ind_loading_sig ~ beta(1.2, 2);
  
  st_int_fact_sig ~ beta(1.2, 2);//one for each series
  ind_int_fact_sig ~ beta(1.2,2);//one for each series
  
  intercept_e_sig ~ beta(1.2, 2);//one for each series

  
 
  for(ser_i in 1:N_series){
    
    sig_hyper_sig[ser_i] ~ cauchy(2,2)T[0,];//one for each series
    sig_hyper_mu[ser_i] ~ cauchy(0,2)T[0,];//one for each series
    //measurement error sig
    for(st_i in 1:N_states)
      for(ind_i in 1:N_industries)
        sig_ID_st_ind[ser_i][st_i,ind_i] ~ cauchy(sig_hyper_mu[ser_i], sig_hyper_sig[ser_i])T[0,];
 
    
    //loadings
    for(st_i in 1:N_states){

      macro_loading[ser_i][st_i] ~ normal(1, macro_loading_sig)T[0,];
            
      for(ind_i in 1:N_industries){
        st_ind_loading_on_ind_process[ser_i][st_i, ind_i] ~ 
          normal(1,ind_loading_sig)T[0,];    

        st_ind_loading_on_st_process[ser_i][st_i,ind_i] ~ normal(1, st_loading_sig)T[0,];
      }
    }
    
    
    //intercept parameters
    st_int_fact[ser_i] ~ normal(0, st_int_fact_sig[ser_i]*2);
    ind_int_fact[ser_i] ~ normal(0, ind_int_fact_sig[ser_i]*2);
    for(st_i in 1:N_states)
      intercept_e_ID_st_ind[ser_i][st_i] ~ normal(0, intercept_e_sig[ser_i]*2);
    
    
  
    
    
    //coeff for change in emp growth
    for(st_i in 1:N_states)
      for(ind_i in 1:N_industries)
        beta_emp_change_ID_st_ind[ser_i][st_i,ind_i] ~ normal(beta_emp_change_mu0[ser_i][ind_i],
                                                       beta_emp_change_sig0[ser_i]);
    
    //one for each industry
    beta_emp_change_mu0[ser_i] ~ normal(beta_emp_change_mu00[ser_i], beta_emp_change_sig00[ser_i]*2);
    beta_emp_change_mu00[ser_i] ~ normal(0,10);
    beta_emp_change_sig00[ser_i] ~ beta(1.1, 1.1);

    //one common to all industries
    beta_emp_change_sig0[ser_i] ~ beta(1.1, 2);



    //coeff for growth in emp growth
    for(st_i in 1:N_states)
      for(ind_i in 1:N_industries)
        beta_emp_growth_ID_st_ind[ser_i][st_i,ind_i] ~ normal(beta_emp_growth_mu0[ser_i][ind_i],
                                                              beta_emp_growth_sig0[ser_i]);
    
    //one for each industry
    beta_emp_growth_mu0[ser_i] ~ normal(beta_emp_growth_mu00[ser_i], beta_emp_growth_sig00[ser_i]*2);
    beta_emp_growth_mu00[ser_i] ~ normal(0,10);
    beta_emp_growth_sig00[ser_i] ~ beta(1.1, 1.1);

    //one common to all industries
    beta_emp_growth_sig0[ser_i] ~ beta(1.1, 2);

    
  }



  {//likelihood
      
     matrix[N_series, N_series] L;
     L <- cholesky_decompose(e_Omega);
 
 
 
    for(obs_i in 1:N_obs){
      int st_i;
      int yr_i;
      int ind_i;    

      vector[N_series] mu_i;
      matrix[N_series,N_series] L_sig_i;
      
      st_i <- obs_state_map[obs_i];
      yr_i <- obs_year_map[obs_i];
      ind_i <- obs_industry_map[obs_i];
      
      for(ser_i in 1:N_series)
        mu_i[ser_i] <- intercept_ID_st_ind[ser_i][st_i, ind_i] + 
                     change_in_emp_growth_ID_st_ind_yr[st_i, ind_i, yr_i] * beta_emp_change_ID_st_ind[ser_i][st_i, ind_i] +
                     
                     emp_growth_ID_st_ind_yr[st_i, ind_i, yr_i] * beta_emp_growth_ID_st_ind[ser_i][st_i, ind_i] +
                     macro_hidden[yr_i][ser_i] * macro_loading[ser_i][st_i] +
                     st_hidden[st_i,yr_i][ser_i] * st_ind_loading_on_st_process[ser_i][st_i,ind_i] +
                     ind_hidden[ind_i,yr_i][ser_i] * st_ind_loading_on_ind_process[ser_i][st_i,ind_i];


      for (s1 in 1:N_series)
        for (s2 in 1:s1)
          L_sig_i[s1,s2] <- sig_ID_st_ind[s1][st_i, ind_i] * L[s1,s2];
      

      obs_growth[obs_i] ~ multi_normal_cholesky(mu_i, L_sig_i);
    }
   
  }

 
}

generated quantities{

  vector[N_series] macro_hidden_full[N_obs_years + N_future_years];
  vector[N_series] ind_hidden_full[N_industries, N_obs_years + N_future_years];
  vector[N_series] st_hidden_full[N_states, N_obs_years + N_future_years];
  



  for(ser_i in 1:N_series){
    
    for(yr_i in 1:N_obs_years){
      macro_hidden_full[yr_i][ser_i] <- macro_hidden[yr_i][ser_i];
      
      for(st_i in 1:N_states)
        st_hidden_full[st_i,yr_i][ser_i] <- st_hidden[st_i,yr_i][ser_i];
      
      for(ind_i in 1:N_industries)
        ind_hidden_full[ind_i,yr_i][ser_i] <- ind_hidden[ind_i,yr_i][ser_i];
    }

  }
  

  {
    matrix[N_series, N_series] S;

    S <- macro_hidden_innov_sig*macro_hidden_innov_sig *
       macro_Omega;
    
    for(yr_i in (N_obs_years+1):(N_obs_years+N_future_years))
      macro_hidden_full[yr_i] <- multi_normal_rng(rho .* macro_hidden_full[yr_i-1],S);
  }

  {
    matrix[N_series, N_series] S;

    S <- st_hidden_innov_sig * st_hidden_innov_sig *
       st_Omega;
    
    for(yr_i in (N_obs_years+1):(N_obs_years+N_future_years))
      for(st_i in 1:N_states)
        st_hidden_full[st_i,yr_i] <- multi_normal_rng(st_hidden_full[st_i,yr_i-1],S);
  }


  {
    matrix[N_series, N_series] S;

    S <- ind_hidden_innov_sig * ind_hidden_innov_sig *
       ind_Omega;
    
    for(yr_i in (N_obs_years+1):(N_obs_years+N_future_years))
      for(ind_i in 1:N_industries)
        ind_hidden_full[ind_i,yr_i] <- multi_normal_rng(rho_ind .* ind_hidden_full[ind_i,yr_i-1],S);
  }


}

