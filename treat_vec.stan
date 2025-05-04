// Hierarchical logistic regression with age and country varying intercepts
data {
  int<lower=1> N;           // Number of observations
  int<lower=1> M;           // Number of predictors
  int y[N];                 // Binary outcome: 1 = seeks treatment, 0 = does not
  matrix[N, M] X;           // Predictor matrix
  int<lower=1> J;           // Number of age groups
  int<lower=1> K;           // Number of countries
  int<lower=1> age_i[N];    // Age-group index for each observation (1,...,J)
  int<lower=1> coun_i[N];   // Country index for each observation (1,...,K)
}

parameters {
  vector[J]    alpha_age;   // Varying intercepts by age group
  vector[K]    alpha_coun;  // Varying intercepts by country
  real         beta_0;      // Global intercept (baseline log-odds)
  vector[M]    beta;        // Regression slopes for predictors (log-odds ratios)
  real<lower=0> sigma_age;  // SD of age-group intercepts
  real<lower=0> sigma_coun; // SD of country intercepts
}

transformed parameters {
  vector[N] pi_;            // log-odds for each observation

  // Build the log-odds for each record by adding:
  // global intercept + X·β + age intercept + country intercept
  for (i in 1:N) {
    pi_[i] = beta_0
           + X[i] * beta
           + alpha_age[age_i[i]]
           + alpha_coun[coun_i[i]];
  }
}

model {
  // --- Likelihood -------------------------------------------------
  // y[n] ~ Bernoulli(inv_logit(pi_[n]))
  target += bernoulli_logit_lpmf(y | pi_);

  // --- Priors for varying intercepts -----------------------------
  // First age-group baseline
  alpha_age[1]     ~ normal(0, 1);
  // Subsequent age groups centered on previous group (random walk)
  alpha_age[2:J]   ~ normal(alpha_age[1:(J-1)], sigma_age);

  // All country intercepts drawn from a common Normal(0, sigma_coun)
  alpha_coun[1:K]  ~ normal(0, sigma_coun);

  // --- Priors for fixed effects ---------------------------------
  beta_0           ~ normal(0, 1);   // global intercept
  beta             ~ normal(0, 1);   // slopes for predictors

  // --- Priors for hyperparameters -------------------------------
  sigma_age        ~ normal(0, 1);   // SD for age-group effects
  sigma_coun       ~ normal(0, 1);   // SD for country effects
}

generated quantities {
  vector[N] log_lik;    // pointwise log-likelihood (for LOO/WAIC) - diagnostic
  int treat_rep[N];     // posterior predictive replicates of y

  for (n in 1:N) {
    // Recompute linear predictor for each draw
    real lp_n = beta_0
              + X[n] * beta
              + alpha_age[age_i[n]]
              + alpha_coun[coun_i[n]];

    // Save the pointwise log-likelihood
    log_lik[n]   = bernoulli_logit_lpmf(y[n] | lp_n);
    // Generate a posterior predictive observation
    treat_rep[n] = bernoulli_logit_rng(lp_n);
  }
}
