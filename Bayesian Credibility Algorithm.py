"""
This module implements a credibility‑weighted Bayesian mixture model for
longitudinal data with missingness.  The model treats observations from
a primary source (direct measurements) and a collateral source as two
noisy measurements of an underlying latent state for each subject at
each time point.  Missing values in the primary dataset are assumed
MAR (missing at random) and are imputed within a Gibbs sampler using
data augmentation.  The latent state, population mean and variance,
and credibility weights are iteratively updated.  A half‑Cauchy prior
is placed on the latent standard deviation and updated via a
Metropolis–Hastings step.

The main functions provided are:

* simulate_data: simulate a longitudinal dataset with primary and
  collateral measurements, along with monotone MAR missingness.
* gibbs_sampler: run the hybrid Gibbs/MH sampler to estimate the
  posterior distribution of the model parameters and impute missing
  observations.

This code is intended for research and educational use.  It has no
external dependencies beyond numpy and pandas.
"""

import numpy as np
import pandas as pd


def monotone_mar(data: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Generate monotone MAR missingness for a matrix of observations.

    Parameters
    ----------
    data : np.ndarray
        The full data array (n subjects × T time points).  Values will
        be overwritten by NaN according to a dropout process.
    alpha : float
        Intercept of the logit model for missingness.
    beta : float
        Coefficient of the logit model applied to the previously
        observed value.  A positive beta increases the probability of
        dropout following higher observations.

    Returns
    -------
    np.ndarray
        A copy of the input array with some entries set to NaN under
        a monotone dropout mechanism.  Once a value is missing, all
        subsequent values for that subject are set to NaN.
    """
    Y_obs = data.copy()
    n, T = Y_obs.shape
    # Loop over subjects and time points
    for i in range(n):
        for t in range(1, T):
            # If previous is missing, enforce monotone missingness
            if np.isnan(Y_obs[i, t - 1]):
                Y_obs[i, t:] = np.nan
                break
            # Logistic MAR probability based on previous observed value
            logit_p = alpha + beta * Y_obs[i, t - 1]
            p_drop = 1.0 / (1.0 + np.exp(-logit_p))
            if np.random.rand() < p_drop:
                Y_obs[i, t:] = np.nan
                break
    return Y_obs


def simulate_data(
    N_primary: int,
    N_collateral: int,
    T: int,
    mu: float = 0.0,
    tau: float = 1.0,
    phi: float = 0.0,
    sigma_primary: float = 1.0,
    sigma_collateral: float = 0.6,
    alpha_primary: float = -1.0,
    beta_primary: float = 1.0,
    alpha_collateral: float = -2.0,
    beta_collateral: float = 0.5,
) -> dict:
    """Simulate primary and collateral longitudinal data with MAR missingness.

    A latent AR(1) trajectory is generated for each subject.  Primary
    measurements are sampled from the latent state plus Gaussian noise.
    Collateral measurements are sampled from the same latent state but
    with lower variance.  Missingness is introduced to the primary
    dataset using a monotone MAR mechanism that depends on the
    previous observed value.

    Parameters
    ----------
    N_primary : int
        Number of subjects in the primary dataset.
    N_collateral : int
        Number of subjects in the collateral dataset (can be larger or
        equal to N_primary).
    T : int
        Number of time points.
    mu : float, optional
        Mean of the latent state distribution.  Defaults to 0.
    tau : float, optional
        Standard deviation of the latent state distribution.  Defaults
        to 1.
    phi : float, optional
        AR(1) autocorrelation coefficient for latent trajectories.  A
        value of 0 yields independent trajectories.  Defaults to 0.
    sigma_primary : float, optional
        Standard deviation of the observation error for the primary
        dataset.  Defaults to 1.
    sigma_collateral : float, optional
        Standard deviation of the observation error for the collateral
        dataset.  Defaults to 0.6.
    alpha_primary, beta_primary : float, optional
        Parameters of the logistic MAR mechanism for the primary
        dataset.  Larger beta produces missingness conditional on
        previous values.  Defaults to (-1, 1).
    alpha_collateral, beta_collateral : float, optional
        Parameters of the logistic MAR mechanism for the collateral
        dataset.  Defaults to (-2, 0.5).

    Returns
    -------
    dict
        A dictionary containing the following keys:

        * 'theta_primary': latent states for primary subjects
        * 'theta_collateral': latent states for collateral subjects
        * 'Y_full': complete primary observations
        * 'C_full': complete collateral observations
        * 'Y_obs': primary observations with missingness
        * 'C_obs': collateral observations with missingness
        * 'mask_primary': boolean mask of missing entries in Y_obs
    """
    # Simulate latent AR(1) trajectories
    def simulate_latent(n, T, mu, tau, phi):
        theta = np.zeros((n, T))
        theta[:, 0] = np.random.normal(mu, tau, size=n)
        for t in range(1, T):
            theta[:, t] = phi * theta[:, t - 1] + np.random.normal(mu, tau, size=n)
        return theta

    theta_primary = simulate_latent(N_primary, T, mu, tau, phi)
    theta_collateral = simulate_latent(N_collateral, T, mu, tau, phi)

    # Generate observations
    Y_full = theta_primary + np.random.normal(0.0, sigma_primary, size=(N_primary, T))
    C_full = theta_collateral + np.random.normal(0.0, sigma_collateral, size=(N_collateral, T))

    # Introduce monotone MAR missingness
    Y_obs = monotone_mar(Y_full, alpha_primary, beta_primary)
    C_obs = monotone_mar(C_full, alpha_collateral, beta_collateral)

    mask_primary = np.isnan(Y_obs)

    return {
        "theta_primary": theta_primary,
        "theta_collateral": theta_collateral,
        "Y_full": Y_full,
        "C_full": C_full,
        "Y_obs": Y_obs,
        "C_obs": C_obs,
        "mask_primary": mask_primary,
    }


def sample_half_cauchy(scale: float) -> float:
    """Draw a single sample from a half‑Cauchy distribution.

    The half‑Cauchy distribution has density

        p(x) = (2 / (π * scale)) * 1/(1 + (x/scale)^2),  for x > 0.

    Parameters
    ----------
    scale : float
        Scale parameter of the half‑Cauchy distribution.

    Returns
    -------
    float
        A positive sample drawn from the half‑Cauchy distribution.
    """
    x = np.random.standard_cauchy()
    return abs(x) * scale


def gibbs_sampler(
    Y_obs: np.ndarray,
    C: np.ndarray,
    mask: np.ndarray,
    sigma_y: float,
    sigma_c: float,
    mu0: float,
    sigma0_sq: float,
    tau_scale: float,
    iterations: int = 2000,
    burn_in: int = 1000,
    Z_fixed: float | None = None,
    a_z: float = 1.0,
    b_z: float = 1.0,
    prop_sd_tau: float = 0.1,
    random_state: int | None = None,
) -> dict:
    """Run a hybrid Gibbs/MH sampler for the credibility‑weighted model.

    Parameters
    ----------
    Y_obs : np.ndarray
        Primary data with missing entries (NaN).  Shape (n, T).
    C : np.ndarray
        Collateral data (no missing values) aligned to the primary
        subjects.  Shape (n, T).  If the collateral dataset has more
        subjects than the primary, only the first n rows are used.
    mask : np.ndarray
        Boolean mask with True where Y_obs is missing.
    sigma_y : float
        Observation standard deviation for Y.
    sigma_c : float
        Observation standard deviation for C.
    mu0 : float
        Prior mean for the global mean μ.
    sigma0_sq : float
        Prior variance for μ.
    tau_scale : float
        Scale parameter of the half‑Cauchy prior on τ.
    iterations : int, optional
        Total number of MCMC iterations to run.  Defaults to 2000.
    burn_in : int, optional
        Number of initial iterations to discard as burn‑in.  Defaults
        to 1000.  Burn‑in should be less than iterations.
    Z_fixed : float or None, optional
        If provided, the credibility weight Z is fixed to this value
        for all observations.  Otherwise Z is sampled from its
        posterior using a Beta(a_z, b_z) proposal with Metropolis
        acceptance.
    a_z, b_z : float, optional
        Parameters of the Beta prior for the credibility weight Z
        when Z is sampled.  Defaults to (1,1) corresponding to a
        uniform prior.
    prop_sd_tau : float, optional
        Standard deviation of the random walk proposal for τ in the
        Metropolis–Hastings step.  Defaults to 0.1.
    random_state : int or None, optional
        Seed for the random number generator.

    Returns
    -------
    dict
        A dictionary containing posterior summaries:

        * 'theta_mean': posterior mean of the latent states θ
        * 'Y_mean': posterior mean of the imputed Y values
        * 'mu_chain': chain of μ values after burn‑in
        * 'tau_chain': chain of τ values after burn‑in
        * 'Z_mean': posterior mean of Z (if sampled)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n, T = Y_obs.shape

    # Ensure C has same shape; truncate if necessary
    if C.shape[0] > n:
        C = C[:n, :]

    # Initialize Y_curr by imputing missing values with column means
    Y_curr = Y_obs.copy()
    col_means = np.nanmean(Y_curr, axis=0)
    inds = np.where(np.isnan(Y_curr))
    Y_curr[inds] = np.take(col_means, inds[1])

    # Initialize parameters
    theta = Y_curr.copy()  # start latent states at observed/imputed values
    mu = mu0
    tau = sample_half_cauchy(tau_scale)

    # Initialize Z
    if Z_fixed is not None:
        Z = np.full((n, T), Z_fixed)
    else:
        Z = np.full((n, T), 0.5)

    # Accumulators for posterior means
    theta_sum = np.zeros_like(theta)
    Y_sum = np.zeros_like(Y_curr)
    Z_sum = np.zeros_like(Z)
    samples_collected = 0
    mu_chain = []
    tau_chain = []

    # Pre‑compute constants for sigma_y and sigma_c reciprocals
    inv_sigma_y_sq = 1.0 / (sigma_y ** 2)
    inv_sigma_c_sq = 1.0 / (sigma_c ** 2)

    for s in range(iterations):
        # -----------------------------------------------------------------
        # Step 1: Impute missing primary observations
        # -----------------------------------------------------------------
        for i in range(n):
            for t in range(T):
                if mask[i, t]:
                    # Draw from N(theta_i,t, sigma_y^2)
                    Y_curr[i, t] = np.random.normal(theta[i, t], sigma_y)

        # -----------------------------------------------------------------
        # Step 2: Update latent states theta
        # -----------------------------------------------------------------
        for i in range(n):
            for t in range(T):
                zi = Z[i, t]
                # Primary contribution only if observed
                Ri = 1.0 if not mask[i, t] else 0.0
                # Precision and mean calculations
                prec = (1.0 / (tau ** 2)) + zi * Ri * inv_sigma_y_sq + (1.0 - zi) * inv_sigma_c_sq
                mean_num = (mu / (tau ** 2))
                if not mask[i, t]:
                    mean_num += zi * Ri * Y_curr[i, t] * inv_sigma_y_sq
                # Always include collateral component
                mean_num += (1.0 - zi) * C[i, t] * inv_sigma_c_sq
                var = 1.0 / prec
                theta[i, t] = np.random.normal(mean_num * var, np.sqrt(var))

        # -----------------------------------------------------------------
        # Step 3: Update credibility weights Z (if sampled)
        # -----------------------------------------------------------------
        if Z_fixed is None:
            # Update Z for each subject and time point
            for i in range(n):
                for t in range(T):
                    # If primary observation is missing, Z has no effect
                    # on the primary likelihood and its posterior weight is
                    # determined solely by the collateral likelihood.
                    # We still update Z to reflect Beta prior influence.
                    # Compute the log‑likelihood difference Δ = log fY - log fC.
                    # For missing Y (Ri=0), Δ is negative: fY contributes
                    # nothing, so Δ = -log fC.
                    diff_y = Y_curr[i, t] - theta[i, t]
                    diff_c = C[i, t] - theta[i, t]
                    log_fy = -0.5 * diff_y ** 2 * inv_sigma_y_sq
                    log_fc = -0.5 * diff_c ** 2 * inv_sigma_c_sq
                    delta = log_fy - log_fc
                    z_old = Z[i, t]
                    # Propose new Z from Beta(a_z, b_z)
                    z_new = np.random.beta(a_z, b_z)
                    # Compute Metropolis acceptance ratio
                    log_r = (z_new - z_old) * delta
                    if np.log(np.random.rand()) < log_r:
                        Z[i, t] = z_new
        # -----------------------------------------------------------------
        # Step 4: Update global mean mu
        # -----------------------------------------------------------------
        theta_vec = theta.ravel()
        post_prec = (len(theta_vec) / (tau ** 2)) + (1.0 / sigma0_sq)
        post_mean = ((np.sum(theta_vec) / (tau ** 2)) + (mu0 / sigma0_sq)) / post_prec
        mu = np.random.normal(post_mean, np.sqrt(1.0 / post_prec))

        # -----------------------------------------------------------------
        # Step 5: Update latent variance tau via MH
        # -----------------------------------------------------------------
        # Propose new tau via symmetric random walk on the positive reals
        tau_prop = abs(tau + np.random.normal(scale=prop_sd_tau))
        if tau_prop == 0:
            tau_prop = 1e-6
        # Compute log posterior for current and proposed tau (up to constant)
        # Log prior: proportional to -log(1 + (tau/scale)^2)
        def log_post_tau(t):
            return (
                - (n * T) * np.log(t)
                - np.sum((theta - mu) ** 2) / (2.0 * (t ** 2))
                - np.log(1.0 + (t / tau_scale) ** 2)
            )
        logp_curr = log_post_tau(tau)
        logp_prop = log_post_tau(tau_prop)
        log_alpha = logp_prop - logp_curr
        if np.log(np.random.rand()) < log_alpha:
            tau = tau_prop

        # -----------------------------------------------------------------
        # After burn‑in, accumulate results
        # -----------------------------------------------------------------
        if s >= burn_in:
            theta_sum += theta
            Y_sum += Y_curr
            if Z_fixed is None:
                Z_sum += Z
            mu_chain.append(mu)
            tau_chain.append(tau)
            samples_collected += 1

    # Compute posterior means
    theta_mean = theta_sum / samples_collected
    Y_mean = Y_sum / samples_collected
    result = {
        "theta_mean": theta_mean,
        "Y_mean": Y_mean,
        "mu_chain": np.array(mu_chain),
        "tau_chain": np.array(tau_chain),
    }
    if Z_fixed is None:
        result["Z_mean"] = Z_sum / samples_collected
    return result


if __name__ == "__main__":
    # Example usage: simulate data and run sampler
    sim = simulate_data(
        N_primary=100,
        N_collateral=150,
        T=3,
        mu=0.0,
        tau=1.0,
        phi=0.0,
        sigma_primary=1.0,
        sigma_collateral=0.6,
        alpha_primary=-1.2,
        beta_primary=0.8,
        alpha_collateral=-2.0,
        beta_collateral=0.5,
    )
    Y_obs = sim["Y_obs"]
    C_obs = sim["C_obs"][: Y_obs.shape[0], :]
    mask = sim["mask_primary"]
    # Run sampler with sampled Z
    result = gibbs_sampler(
        Y_obs,
        C_obs,
        mask,
        sigma_y=1.0,
        sigma_c=0.6,
        mu0=0.0,
        sigma0_sq=25.0,
        tau_scale=2.5,
        iterations=1500,
        burn_in=500,
        Z_fixed=None,
        a_z=1.0,
        b_z=1.0,
        prop_sd_tau=0.05,
        random_state=42,
    )
    # Print summary statistics
    print("Posterior mean of mu:", np.mean(result["mu_chain"]))
    print("Posterior mean of tau:", np.mean(result["tau_chain"]))
    if "Z_mean" in result:
        print("Posterior mean of Z (first few entries):")
        print(result["Z_mean"][:5, :])


        