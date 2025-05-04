// Hierarchical logistic regression model
data {
  int<lower=1> N;       // Number of observations
  int<lower=1> M;       // Number of predictors
  int y[N];             // Binary outcome: 1 = seeks treatment, 0 = does not
  matrix[N, M] X;       // Predictor matrix
}

parameters {
  real beta_0;          // Intercept (baseline log‑odds of seeking treatment)
  vector[M] beta;       // Regression coefficients for each of the M predictors (log-odds ratio)
}

transformed parameters {
  vector[N] pi_;        // log-odds for each observation

  // Compute log-odds for each observation:
  for (i in 1:N) {
    // beta_0 + dot_product of row i of X with beta vector
    pi_[i] = beta_0 + X[i] * beta;
  }
}

model {
  // --- Likelihood -------------------------------------------------
  // Increment the log-probability target by the log-PMF of y under
  // a Bernoulli with logit(pi_)
  target += bernoulli_logit_lpmf(y | pi_);

  // --- Priors -----------------------------------------------------
  // Weakly informative normal(0, 1) priors on intercept and slopes:
  beta_0 ~ normal(0, 1);
  beta   ~ normal(0, 1);
}

generated quantities {
  vector[N] log_lik;    // Pointwise log-likelihood for each data point (for LOO/WAIC)
  int treat_rep[N];     // Posterior predictive draws for y

  for (n in 1:N) {
    // Recompute linear predictor for nth observation
    real pi_post_n = beta_0 + X[n] * beta;

    // Save pointwise log-likelihood
    log_lik[n] = bernoulli_logit_lpmf(y[n] | pi_post_n);

    // Draw a replicated outcome from the posterior predictive
    treat_rep[n] = bernoulli_logit_rng(pi_post_n);
  }
}
