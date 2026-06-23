# ============================================================
# SCENARIO 4: AR(1) STRUCTURE WITH SAMPLED Z
# Credibility-weighted Bayesian machine-learning imputation
#
# This script modifies the AR(1) fixed-Z scenario to use
# sampled credibility weights under AR(1) latent trajectories:
#
#   theta_i1 ~ N(mu, tau^2)
#   theta_it | theta_i,t-1 ~ N(mu + phi(theta_i,t-1 - mu), tau^2)
#
# Sampled Z priors considered:
#   Z_t ~ Beta(1,1), Beta(2,2), Beta(2,3), or Beta(3,2)
#
# Priors:
#   mu  ~ N(0, 5^2) or N(0, 10^2)
#   tau ~ Half-Cauchy(1), Half-Cauchy(2.5), Half-Cauchy(5)
#   phi sampled by MH on Fisher-z scale with uniform prior over (-0.99, 0.99)
#
# Required CSV columns:
#   Direct:     ID, Y_t1, Y_t2, Y_t3
#   Collateral: ID, C_t1, C_t2, C_t3
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from statsmodels.imputation.mice import MICEData
    HAVE_MICE = True
except Exception:
    HAVE_MICE = False

try:
    from sklearn.naive_bayes import GaussianNB
    HAVE_GNB = True
except Exception:
    HAVE_GNB = False


# ============================================================
# 0. User settings
# ============================================================

DIRECT_DATA = r"F:\gacheri's documents\research NCDs\BIBTEX\New folder\Simulated Datasets\direct_dataset_simulated.csv"
COLLATERAL_DATA = r"F:\gacheri's documents\research NCDs\BIBTEX\New folder\Simulated Datasets\collateral_dataset_simulated.csv"

OUTDIR = Path(
    r"F:\gacheri's documents\research NCDs\BIBTEX\New folder\Simulated Datasets\Bayesian Learning Algorithm\AR1_sampledZ_results"
)

PLOTS_DIR = OUTDIR / "graphics"
DIAGNOSTICS_DIR = OUTDIR / "diagnostics"

DISPLAY_PLOTS = False

OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", None)

ID_COL = "ID"
Y_COLS = ["Y_t1", "Y_t2", "Y_t3"]
C_COLS = ["C_t1", "C_t2", "C_t3"]

VALIDATION_FRACTION = 0.20
# Fast screening seeds. For final thesis validation, use:
# [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
VALIDATION_SEEDS = [2026, 2027, 2028]
VALIDATION_SEED = 2026

COLLATERAL_LINK = "first_n"
# If your collateral data are completely external and not subject-paired, use:
# COLLATERAL_LINK = "population_mean"

MU_PRIORS = [(0.0, 5.0), (0.0, 10.0)]
TAU_SCALES = [1.0, 2.5, 5.0]
Z_PRIORS = [
    (1.0, 1.0),   # uniform
    (2.0, 2.0),   # centered around 0.5
    (2.0, 3.0),   # slightly more collateral borrowing
    (3.0, 2.0),   # slightly more primary borrowing
]

# Fast screening MCMC settings.
# For final thesis runs, use N_CHAINS=4, N_ITER=3000 or 5000, and BURN=1000.
N_CHAINS = 2
N_ITER = 1000
BURN = 300

SIGMA_Y_OVERRIDE = None
SIGMA_C_OVERRIDE = None

print("Scenario 4: AR(1) structure with sampled Z")
print("Display plots on screen:", DISPLAY_PLOTS)
print("Validation seeds:", VALIDATION_SEEDS)


# ============================================================
# 1. Load direct and collateral datasets
# ============================================================

direct_df = pd.read_csv(DIRECT_DATA)
collateral_df = pd.read_csv(COLLATERAL_DATA)

missing_y_cols = [col for col in Y_COLS if col not in direct_df.columns]
missing_c_cols = [col for col in C_COLS if col not in collateral_df.columns]

if missing_y_cols:
    raise ValueError(f"Missing direct dataset columns: {missing_y_cols}")

if missing_c_cols:
    raise ValueError(f"Missing collateral dataset columns: {missing_c_cols}")

if ID_COL in direct_df.columns and ID_COL in collateral_df.columns:
    model_df = direct_df[[ID_COL] + Y_COLS].merge(
        collateral_df[[ID_COL] + C_COLS],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    if model_df[C_COLS].isna().all(axis=1).any():
        bad_ids = model_df.loc[
            model_df[C_COLS].isna().all(axis=1),
            ID_COL
        ].head().tolist()
        raise ValueError(
            "Some direct IDs have no matching collateral row. "
            f"Example missing IDs: {bad_ids}"
        )

    direct_model_df = direct_df.loc[
        direct_df[ID_COL].isin(model_df[ID_COL])
    ].copy()
    subject_ids = model_df[ID_COL].to_numpy()
else:
    n_align = min(len(direct_df), len(collateral_df))
    model_df = pd.concat(
        [
            direct_df.loc[: n_align - 1, Y_COLS].reset_index(drop=True),
            collateral_df.loc[: n_align - 1, C_COLS].reset_index(drop=True),
        ],
        axis=1,
    )
    direct_model_df = direct_df.iloc[:n_align].copy()
    subject_ids = np.arange(1, n_align + 1)

Y_obs_raw = model_df[Y_COLS].to_numpy(dtype=float)
C_obs_raw = model_df[C_COLS].to_numpy(dtype=float)

N, T = Y_obs_raw.shape
mask_miss = np.isnan(Y_obs_raw)
R = (~mask_miss).astype(float)

if T < 2:
    raise ValueError("AR(1) structure requires at least two timepoints.")

if mask_miss.sum() == 0:
    raise ValueError(
        "Y_primary_obs has no missing entries. Imputation RMSE and coverage cannot be computed."
    )

observed_positions = np.argwhere(~mask_miss)


def make_validation_split(validation_seed):
    """Create one validation holdout split from observed Y values."""
    rng_validation = np.random.default_rng(validation_seed)
    n_validation = max(1, int(round(VALIDATION_FRACTION * len(observed_positions))))

    validation_choice = rng_validation.choice(
        len(observed_positions),
        size=n_validation,
        replace=False,
    )

    validation_mask = np.zeros_like(mask_miss, dtype=bool)
    for i_val, t_val in observed_positions[validation_choice]:
        validation_mask[i_val, t_val] = True

    validation_true = Y_obs_raw.copy()
    model_input = Y_obs_raw.copy()
    model_input[validation_mask] = np.nan
    model_mask = np.isnan(model_input)
    r_model = (~model_mask).astype(float)

    return validation_mask, validation_true, model_input, model_mask, r_model


mask_validation, Y_validation_true, Y_model_input, mask_model, R_model = make_validation_split(
    VALIDATION_SEED
)

# Truth is not assumed available when using already-simulated CSV files.
Y_true_available = False
theta_true_available = False
mu_true_available = False
tau_true_available = False
phi_true_available = False

Y_true = None
theta_true = None
mu_true = np.nan
tau_true = np.nan
phi_true = np.nan

SIGMA_Y = float(np.nanstd(Y_obs_raw, ddof=1))
SIGMA_C = float(np.nanstd(C_obs_raw, ddof=1))

if SIGMA_Y_OVERRIDE is not None:
    SIGMA_Y = float(SIGMA_Y_OVERRIDE)

if SIGMA_C_OVERRIDE is not None:
    SIGMA_C = float(SIGMA_C_OVERRIDE)

if not np.isfinite(SIGMA_Y) or SIGMA_Y <= 0:
    SIGMA_Y = 1.0

if not np.isfinite(SIGMA_C) or SIGMA_C <= 0:
    SIGMA_C = max(0.6 * SIGMA_Y, 1e-6)

SIGMA_Y_GRID = [0.8 * SIGMA_Y, SIGMA_Y, 1.2 * SIGMA_Y]
SIGMA_C_GRID = [0.5 * SIGMA_Y, 0.6 * SIGMA_Y, 0.7 * SIGMA_Y, 0.8 * SIGMA_Y]

print("Direct data:", DIRECT_DATA)
print("Collateral data:", COLLATERAL_DATA)
print("Output directory:", OUTDIR)
print("Primary shape:", Y_obs_raw.shape)
print("Collateral shape:", C_obs_raw.shape)
print("Missing primary entries:", int(mask_miss.sum()))
print("Validation holdout entries:", int(mask_validation.sum()))
print("sigma_y used:", SIGMA_Y)
print("sigma_c used:", SIGMA_C)
print("sigma_y tuning grid:", SIGMA_Y_GRID)
print("sigma_c tuning grid:", SIGMA_C_GRID)


# ============================================================
# 2. Helper functions
# ============================================================

def fill_column_means(A):
    A = np.asarray(A, dtype=float).copy()
    col_means = np.nanmean(A, axis=0)
    grand_mean = np.nanmean(A)

    if np.isnan(grand_mean):
        grand_mean = 0.0

    col_means = np.where(np.isnan(col_means), grand_mean, col_means)
    inds = np.where(np.isnan(A))
    A[inds] = np.take(col_means, inds[1])

    return A


def align_collateral(C_obs, n_primary, T, method="first_n"):
    C_filled = fill_column_means(C_obs)

    if C_filled.shape[1] != T:
        raise ValueError(
            "Collateral data must have the same number of timepoints as primary data."
        )

    if method == "first_n":
        if C_filled.shape[0] < n_primary:
            reps = int(np.ceil(n_primary / C_filled.shape[0]))
            C_filled = np.tile(C_filled, (reps, 1))
        return C_filled[:n_primary, :]

    if method == "population_mean":
        cbar = np.nanmean(C_filled, axis=0)
        return np.tile(cbar, (n_primary, 1))

    raise ValueError("method must be 'first_n' or 'population_mean'.")


def rhat(chains):
    x = np.asarray(chains, dtype=float)

    if x.ndim != 2 or x.shape[0] < 2:
        return np.nan

    m, n = x.shape
    chain_means = x.mean(axis=1)
    chain_vars = x.var(axis=1, ddof=1)

    B = n * chain_means.var(ddof=1)
    W = chain_vars.mean()

    if W <= 0:
        return np.nan

    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W))


def fixed_z_from_missingness(missing_fraction):
    if missing_fraction <= 0.10:
        return 0.90
    if missing_fraction <= 0.30:
        return 0.70
    return 0.50


def log_target_eta_tau_ar1(eta, theta, mu_current, phi_current, tau_scale):
    """
    Log target for eta = log(tau) under AR(1) latent structure.

    theta_i1 ~ N(mu, tau^2)
    theta_it | theta_i,t-1 ~ N(mu + phi(theta_i,t-1 - mu), tau^2)

    tau ~ Half-Cauchy(tau_scale)
    """
    tau_value = np.exp(eta)

    resid0 = theta[:, 0] - mu_current

    if theta.shape[1] > 1:
        mean_trans = mu_current + phi_current * (theta[:, :-1] - mu_current)
        resid_trans = theta[:, 1:] - mean_trans
        ss = np.sum(resid0**2) + np.sum(resid_trans**2)
    else:
        ss = np.sum(resid0**2)

    n_terms = theta.size

    log_lik = -n_terms * np.log(tau_value) - ss / (2.0 * tau_value**2)
    log_prior = -np.log1p((tau_value / tau_scale) ** 2)
    log_jacobian = eta

    return log_lik + log_prior + log_jacobian


def log_target_eta_phi_ar1(eta_phi, theta, mu_current, tau_current):
    """
    Log target for eta_phi = atanh(phi).

    We use phi = tanh(eta_phi), giving phi in (-1, 1).
    A uniform prior over phi is assumed. The Jacobian log(1 - phi^2)
    is included because the proposal is on eta_phi.
    """
    phi_value = np.tanh(eta_phi)

    mean_trans = mu_current + phi_value * (theta[:, :-1] - mu_current)
    resid_trans = theta[:, 1:] - mean_trans

    log_lik = -np.sum(resid_trans**2) / (2.0 * tau_current**2)
    log_jacobian = np.log(max(1.0 - phi_value**2, 1e-12))

    return log_lik + log_jacobian


def sample_mu_ar1(theta, tau_current, phi_current, mu0, sigma0, rng):
    """
    Gibbs update for mu under centered AR(1):

        theta_i1 ~ N(mu, tau^2)
        theta_it - phi theta_i,t-1 ~ N((1 - phi)mu, tau^2)

    Prior:
        mu ~ N(mu0, sigma0^2)
    """
    n, T = theta.shape
    tau2 = tau_current**2
    sigma0_sq = sigma0**2

    precision = 1.0 / sigma0_sq
    numerator = mu0 / sigma0_sq

    # Initial state contribution
    precision += n / tau2
    numerator += np.sum(theta[:, 0]) / tau2

    # Transition contributions
    if T > 1:
        d = theta[:, 1:] - phi_current * theta[:, :-1]
        k = 1.0 - phi_current

        precision += (k**2) * d.size / tau2
        numerator += k * np.sum(d) / tau2

    post_var = 1.0 / precision
    post_mean = post_var * numerator

    return rng.normal(post_mean, np.sqrt(post_var))


def compute_metrics(y_hat, y_true, mask):
    diff = y_hat[mask] - y_true[mask]
    return {
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "Bias": float(np.mean(diff)),
        "SE": float(np.std(diff, ddof=1)) if diff.size > 1 else np.nan,
    }


def compute_validation_metrics(y_hat, y_true, validation_mask):
    diff = y_hat[validation_mask] - y_true[validation_mask]
    return {
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "Bias": float(np.mean(diff)),
        "SE": float(np.std(diff, ddof=1)) if diff.size > 1 else np.nan,
    }


def validation_draw_columns(model_mask, validation_mask):
    positions = list(zip(*np.where(model_mask)))
    pos_to_col = {pos: col for col, pos in enumerate(positions)}
    return np.array(
        [pos_to_col[pos] for pos in zip(*np.where(validation_mask))],
        dtype=int,
    )


def coverage_from_draws(draws, y_true, model_mask, validation_mask):
    cols = validation_draw_columns(model_mask, validation_mask)
    eval_draws = draws[:, cols]
    truth = y_true[validation_mask]

    lower = np.quantile(eval_draws, 0.025, axis=0)
    upper = np.quantile(eval_draws, 0.975, axis=0)

    return float(np.mean((truth >= lower) & (truth <= upper)))


def coverage_from_chain_outputs(chain_outputs, y_true, model_mask, validation_mask):
    cols = validation_draw_columns(model_mask, validation_mask)
    eval_draws = np.vstack(
        [o["Y_miss_draws"][:, cols] for o in chain_outputs]
    )
    truth = y_true[validation_mask]

    lower = np.quantile(eval_draws, 0.025, axis=0)
    upper = np.quantile(eval_draws, 0.975, axis=0)

    return float(np.mean((truth >= lower) & (truth <= upper)))


def save_and_maybe_show(fig, output_path, display_plots=DISPLAY_PLOTS):
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if display_plots:
        plt.show()
    plt.close(fig)


def safe_filename(text):
    return (
        str(text)
        .replace(" ", "_")
        .replace(".", "p")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
    )


def write_table_section(handle, title, df, index=False):
    handle.write(f"\n{title}\n")
    handle.write("=" * len(title) + "\n")

    if df is None or df.empty:
        handle.write("No rows available.\n")
    else:
        handle.write(df.to_string(index=index))
        handle.write("\n")


def iterative_regression_impute(Y_obs, n_updates=10):
    Y_imp = fill_column_means(Y_obs)
    missing = np.isnan(Y_obs)

    for _ in range(n_updates):
        for t in range(Y_obs.shape[1]):
            miss_t = missing[:, t]
            obs_t = ~miss_t

            if miss_t.sum() == 0 or obs_t.sum() < 3:
                continue

            predictors = [j for j in range(Y_obs.shape[1]) if j != t]
            X_obs = np.column_stack(
                [np.ones(obs_t.sum()), Y_imp[obs_t][:, predictors]]
            )
            y_obs = Y_imp[obs_t, t]

            try:
                beta = np.linalg.lstsq(X_obs, y_obs, rcond=None)[0]
                X_miss = np.column_stack(
                    [np.ones(miss_t.sum()), Y_imp[miss_t][:, predictors]]
                )
                Y_imp[miss_t, t] = X_miss @ beta
            except np.linalg.LinAlgError:
                Y_imp[miss_t, t] = np.nanmean(y_obs)

    return Y_imp


def naive_mean_impute(Y_obs):
    return fill_column_means(Y_obs)


def mice_impute(Y_obs, n_updates=10):
    if not HAVE_MICE:
        return iterative_regression_impute(Y_obs, n_updates=n_updates)

    try:
        df = pd.DataFrame(
            Y_obs,
            columns=[f"t{j+1}" for j in range(Y_obs.shape[1])]
        )

        imp = MICEData(df)

        for _ in range(n_updates):
            imp.update_all()

        return imp.data.to_numpy(dtype=float)

    except Exception as e:
        print("MICE failed:", e)
        return iterative_regression_impute(Y_obs, n_updates=n_updates)


def gaussian_nb_predict_numpy(X_train, y_train, X_test):
    classes = np.unique(y_train)
    log_probs = []

    for cls in classes:
        X_cls = X_train[y_train == cls]
        prior = X_cls.shape[0] / X_train.shape[0]
        mean = X_cls.mean(axis=0)
        var = X_cls.var(axis=0) + 1e-6
        log_likelihood = -0.5 * (
            np.log(2.0 * np.pi * var)
            + ((X_test - mean) ** 2) / var
        ).sum(axis=1)
        log_probs.append(np.log(prior) + log_likelihood)

    return classes[np.argmax(np.vstack(log_probs).T, axis=1)]


def gaussian_nb_impute(Y_obs, C_aligned):
    Y_imp = fill_column_means(Y_obs)
    Y_initial = fill_column_means(Y_obs)
    X_all = np.column_stack([Y_initial, C_aligned])

    for t in range(Y_obs.shape[1]):
        miss_t = np.isnan(Y_obs[:, t])
        obs_t = ~miss_t

        if miss_t.sum() == 0:
            continue

        y_obs_t = Y_obs[obs_t, t]

        if len(y_obs_t) < 10 or len(np.unique(np.round(y_obs_t, 8))) < 3:
            Y_imp[miss_t, t] = np.nanmean(y_obs_t)
            continue

        try:
            q1, q2 = np.nanquantile(y_obs_t, [1/3, 2/3])

            if q1 == q2:
                Y_imp[miss_t, t] = np.nanmean(y_obs_t)
                continue

            y_class = np.digitize(y_obs_t, bins=[q1, q2])

            if len(np.unique(y_class)) < 2:
                Y_imp[miss_t, t] = np.nanmean(y_obs_t)
                continue

            if HAVE_GNB:
                clf = GaussianNB()
                clf.fit(X_all[obs_t, :], y_class)
                pred_class = clf.predict(X_all[miss_t, :])
            else:
                pred_class = gaussian_nb_predict_numpy(
                    X_train=X_all[obs_t, :],
                    y_train=y_class,
                    X_test=X_all[miss_t, :],
                )

            class_means = {
                k: y_obs_t[y_class == k].mean()
                for k in np.unique(y_class)
            }

            fallback = np.nanmean(y_obs_t)

            Y_imp[miss_t, t] = np.array(
                [class_means.get(k, fallback) for k in pred_class]
            )

        except Exception:
            Y_imp[miss_t, t] = np.nanmean(y_obs_t)

    return Y_imp


def save_missing_posterior_summary(
    y_miss_all,
    y_mean,
    model_mask,
    output_path,
    ids,
    y_cols,
    validation_mask=None,
):
    rows = []
    positions = list(zip(*np.where(model_mask)))

    for col_idx, (i, t) in enumerate(positions):
        draws = y_miss_all[:, col_idx]
        rows.append(
            {
                "ID": ids[i],
                "timepoint": y_cols[t],
                "is_original_missing": bool(mask_miss[i, t]),
                "is_validation_holdout": bool(validation_mask[i, t]) if validation_mask is not None else False,
                "posterior_mean": float(y_mean[i, t]),
                "posterior_se": float(np.std(draws, ddof=1)),
                "ci_2_5": float(np.quantile(draws, 0.025)),
                "ci_97_5": float(np.quantile(draws, 0.975)),
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)



# ============================================================
# 3. AR(1) sampled-Z Gibbs/MH sampler
# ============================================================

def log_normal_density(x, mean, sd):
    """Elementwise Gaussian log-density."""
    return -0.5 * np.log(2.0 * np.pi * sd**2) - 0.5 * ((x - mean) ** 2) / (sd**2)


def update_theta_ar1_sampled_z(
    theta,
    Y_curr,
    C_aligned,
    mask,
    Z,
    mu_current,
    tau_current,
    phi_current,
    sigma_y,
    sigma_c,
    rng,
):
    """
    Component-wise Gibbs update for theta_it under AR(1) prior and time-specific sampled Z_t.

    Observation contribution:
        [fY(Y_it | theta_it)]^(Z_t R_it)
        [fC(C_it | theta_it)]^(1 - Z_t)

    AR(1) latent contribution:
        theta_i1 ~ N(mu, tau^2)
        theta_it | theta_i,t-1 ~ N(mu + phi(theta_i,t-1 - mu), tau^2)
    """
    n, T = theta.shape
    R_local = (~mask).astype(float)

    inv_tau2 = 1.0 / (tau_current**2)
    inv_sigma_y2 = 1.0 / (sigma_y**2)
    inv_sigma_c2 = 1.0 / (sigma_c**2)

    for i in range(n):
        for t in range(T):
            zi = Z[t]
            Ri = R_local[i, t]

            precision = zi * Ri * inv_sigma_y2
            precision += (1.0 - zi) * inv_sigma_c2

            mean_num = (1.0 - zi) * C_aligned[i, t] * inv_sigma_c2

            if Ri == 1.0:
                mean_num += zi * Y_curr[i, t] * inv_sigma_y2

            if t == 0:
                # Initial state: theta_i1 ~ N(mu, tau^2)
                precision += inv_tau2
                mean_num += mu_current * inv_tau2

                # Next-transition contribution: theta_i2 | theta_i1
                if T > 1:
                    a_next = theta[i, t + 1] - mu_current + phi_current * mu_current
                    precision += (phi_current**2) * inv_tau2
                    mean_num += phi_current * a_next * inv_tau2

            elif t == T - 1:
                # Previous-transition contribution: theta_it | theta_i,t-1
                m_prev = mu_current + phi_current * (theta[i, t - 1] - mu_current)
                precision += inv_tau2
                mean_num += m_prev * inv_tau2

            else:
                # Previous-transition contribution
                m_prev = mu_current + phi_current * (theta[i, t - 1] - mu_current)
                precision += inv_tau2
                mean_num += m_prev * inv_tau2

                # Next-transition contribution
                a_next = theta[i, t + 1] - mu_current + phi_current * mu_current
                precision += (phi_current**2) * inv_tau2
                mean_num += phi_current * a_next * inv_tau2

            var = 1.0 / precision
            theta[i, t] = rng.normal(mean_num * var, np.sqrt(var))

    return theta


def update_Z_time_beta_independence_mh(
    Z,
    Y_curr,
    C_aligned,
    theta,
    mask,
    sigma_y,
    sigma_c,
    z_prior,
    rng,
):
    """
    Metropolis-Hastings update for time-specific sampled Z_t.

    Target:
        p(Z_t | rest) ∝ Z_t^(a-1)(1-Z_t)^(b-1) exp(Z_t * sum_i Delta_it)

    where:
        Delta_it = R_it log fY(Y_it | theta_it) - log fC(C_it | theta_it)

    Proposal:
        Z_new ~ Beta(a,b)

    Since the proposal is the same as the Beta prior, prior and proposal
    terms cancel in the MH ratio. Therefore:

        log acceptance = (Z_new,t - Z_old,t) * sum_i Delta_it
    """
    a_z, b_z = z_prior
    R_local = (~mask).astype(float)

    log_fy = log_normal_density(Y_curr, theta, sigma_y)
    log_fc = log_normal_density(C_aligned, theta, sigma_c)

    delta_by_time = np.sum(R_local * log_fy - log_fc, axis=0)

    Z_new = rng.beta(a_z, b_z, size=Z.shape)
    log_accept = (Z_new - Z) * delta_by_time
    log_u = np.log(rng.random(size=Z.shape))

    accept = log_u < log_accept
    Z[accept] = Z_new[accept]

    return Z, int(np.sum(accept)), int(Z.size)


def run_chain_ar1_sampled_z(
    Y_obs,
    C_aligned,
    mask,
    z_prior=(1.0, 1.0),
    mu0=0.0,
    sigma0=5.0,
    tau_scale=2.5,
    n_iter=3000,
    burn=1000,
    sigma_y=1.0,
    sigma_c=0.6,
    proposal_sd_eta_tau=0.08,
    proposal_sd_eta_phi=0.08,
    seed=1234,
):
    rng = np.random.default_rng(seed)

    n, T = Y_obs.shape
    a_z, b_z = z_prior

    Y_curr = fill_column_means(Y_obs)
    theta = Y_curr.copy()
    Z = rng.beta(a_z, b_z, size=T)

    mu_current = mu0

    tau_start = abs(rng.standard_cauchy()) * tau_scale
    tau_start = float(np.clip(tau_start, 0.05, 10.0))
    eta_tau = np.log(tau_start)
    tau_current = tau_start

    phi_start = 0.5
    eta_phi = np.arctanh(np.clip(phi_start, -0.99, 0.99))
    phi_current = float(np.tanh(eta_phi))

    keep = n_iter - burn
    n_miss = int(mask.sum())

    mu_draws = np.zeros(keep)
    tau_draws = np.zeros(keep)
    phi_draws = np.zeros(keep)
    z_mean_draws = np.zeros(keep)
    y_miss_draws = np.zeros((keep, n_miss), dtype=np.float32)

    theta_sum = np.zeros_like(theta)
    Y_sum = np.zeros_like(Y_curr)
    Z_sum = np.zeros_like(Z)

    accept_tau = 0
    accept_phi = 0
    accept_z = 0
    total_z = 0
    keep_idx = 0

    for it in range(n_iter):

        # Step 1: Data augmentation for missing primary Y_it
        Y_curr[mask] = rng.normal(theta[mask], sigma_y)

        # Step 2: Update latent theta_it under AR(1)
        theta = update_theta_ar1_sampled_z(
            theta=theta,
            Y_curr=Y_curr,
            C_aligned=C_aligned,
            mask=mask,
            Z=Z,
            mu_current=mu_current,
            tau_current=tau_current,
            phi_current=phi_current,
            sigma_y=sigma_y,
            sigma_c=sigma_c,
            rng=rng,
        )

        # Step 3: Update time-specific sampled credibility weights Z_t
        Z, acc_z, tot_z = update_Z_time_beta_independence_mh(
            Z=Z,
            Y_curr=Y_curr,
            C_aligned=C_aligned,
            theta=theta,
            mask=mask,
            sigma_y=sigma_y,
            sigma_c=sigma_c,
            z_prior=z_prior,
            rng=rng,
        )
        accept_z += acc_z
        total_z += tot_z

        # Step 4: Update mu | theta, tau, phi
        mu_current = sample_mu_ar1(
            theta=theta,
            tau_current=tau_current,
            phi_current=phi_current,
            mu0=mu0,
            sigma0=sigma0,
            rng=rng,
        )

        # Step 5: Update phi by MH on eta_phi = atanh(phi)
        eta_phi_prop = eta_phi + rng.normal(0.0, proposal_sd_eta_phi)

        log_phi_curr = log_target_eta_phi_ar1(
            eta_phi,
            theta,
            mu_current,
            tau_current,
        )

        log_phi_prop = log_target_eta_phi_ar1(
            eta_phi_prop,
            theta,
            mu_current,
            tau_current,
        )

        if np.log(rng.random()) < (log_phi_prop - log_phi_curr):
            eta_phi = eta_phi_prop
            accept_phi += 1

        phi_current = float(np.tanh(eta_phi))
        phi_current = float(np.clip(phi_current, -0.99, 0.99))
        eta_phi = np.arctanh(phi_current)

        # Step 6: Update tau by MH on eta_tau = log(tau)
        eta_tau_prop = eta_tau + rng.normal(0.0, proposal_sd_eta_tau)

        log_tau_curr = log_target_eta_tau_ar1(
            eta_tau,
            theta,
            mu_current,
            phi_current,
            tau_scale,
        )

        log_tau_prop = log_target_eta_tau_ar1(
            eta_tau_prop,
            theta,
            mu_current,
            phi_current,
            tau_scale,
        )

        if np.log(rng.random()) < (log_tau_prop - log_tau_curr):
            eta_tau = eta_tau_prop
            accept_tau += 1

        tau_current = float(np.exp(eta_tau))

        # Step 7: Store post-burn-in draws
        if it >= burn:
            mu_draws[keep_idx] = mu_current
            tau_draws[keep_idx] = tau_current
            phi_draws[keep_idx] = phi_current
            z_mean_draws[keep_idx] = float(np.mean(Z))
            y_miss_draws[keep_idx, :] = Y_curr[mask]

            theta_sum += theta
            Y_sum += Y_curr
            Z_sum += Z

            keep_idx += 1

    return {
        "mu": mu_draws,
        "tau": tau_draws,
        "phi": phi_draws,
        "Z_mean_chain": z_mean_draws,
        "Y_miss_draws": y_miss_draws,
        "theta_mean": theta_sum / keep,
        "Y_mean": Y_sum / keep,
        "Z_mean_time": Z_sum / keep,
        "Z_mean_matrix": np.tile(Z_sum / keep, (n, 1)),
        "tau_accept_rate": accept_tau / n_iter,
        "phi_accept_rate": accept_phi / n_iter,
        "z_accept_rate": accept_z / max(total_z, 1),
    }


# ============================================================
# 4. Run AR(1) sampled-Z scenario
# ============================================================

C_aligned = align_collateral(
    C_obs_raw,
    N,
    T,
    method=COLLATERAL_LINK,
)

missing_fraction = mask_miss.sum() / Y_obs_raw.size
model_missing_fraction = mask_model.sum() / Y_model_input.size

print("\nPrimary missing fraction:", round(missing_fraction, 4))
print("Model missing fraction including validation holdout:", round(model_missing_fraction, 4))
print("Sampled Z priors:", Z_PRIORS)

all_results = []
model_objects = {}

for z_prior in Z_PRIORS:

    a_z, b_z = z_prior

    for mu0_val, sigma0_val in MU_PRIORS:

        for tau_scale_val in TAU_SCALES:

            for sigma_y_val in SIGMA_Y_GRID:

                for sigma_c_val in SIGMA_C_GRID:

                    label = (
                        f"AR1_SampledZ_Beta{int(a_z)}_{int(b_z)}_"
                        f"muSD{int(sigma0_val)}_"
                        f"HC{str(tau_scale_val).replace('.', 'p')}_"
                        f"sigY{sigma_y_val:.3g}_"
                        f"sigC{sigma_c_val:.3g}"
                    ).replace(".", "p")

                    chain_outputs = []

                    for ch in range(N_CHAINS):

                        out = run_chain_ar1_sampled_z(
                            Y_model_input,
                            C_aligned,
                            mask_model,
                            z_prior=z_prior,
                            mu0=mu0_val,
                            sigma0=sigma0_val,
                            tau_scale=tau_scale_val,
                            n_iter=N_ITER,
                            burn=BURN,
                            sigma_y=sigma_y_val,
                            sigma_c=sigma_c_val,
                            proposal_sd_eta_tau=0.08,
                            proposal_sd_eta_phi=0.08,
                            seed=8026
                            + 1000 * ch
                            + int(100 * a_z)
                            + int(10 * tau_scale_val)
                            + int(round(100 * sigma_y_val))
                            + int(round(1000 * sigma_c_val)),
                        )

                        chain_outputs.append(out)

                    mu_chains = np.vstack([o["mu"] for o in chain_outputs])
                    tau_chains = np.vstack([o["tau"] for o in chain_outputs])
                    phi_chains = np.vstack([o["phi"] for o in chain_outputs])
                    z_mean_chains = np.vstack([o["Z_mean_chain"] for o in chain_outputs])

                    Y_mean = np.mean(
                        [o["Y_mean"] for o in chain_outputs],
                        axis=0,
                    )

                    theta_mean = np.mean(
                        [o["theta_mean"] for o in chain_outputs],
                        axis=0,
                    )

                    Z_mean_matrix = np.mean(
                        [o["Z_mean_matrix"] for o in chain_outputs],
                        axis=0,
                    )
                    Z_mean_time = np.mean(
                        [o["Z_mean_time"] for o in chain_outputs],
                        axis=0,
                    )

                    mu_mean = float(mu_chains.mean())
                    tau_mean = float(tau_chains.mean())
                    phi_mean = float(phi_chains.mean())
                    z_mean = float(z_mean_chains.mean())

                    mu_ci = np.quantile(mu_chains.ravel(), [0.025, 0.975])
                    tau_ci = np.quantile(tau_chains.ravel(), [0.025, 0.975])
                    phi_ci = np.quantile(phi_chains.ravel(), [0.025, 0.975])
                    z_ci = np.quantile(z_mean_chains.ravel(), [0.025, 0.975])

                    row = {
                        "Model": label,
                        "Z_prior": f"Beta({int(a_z)},{int(b_z)})",
                        "a_z": a_z,
                        "b_z": b_z,
                        "mu_prior_sd": sigma0_val,
                        "tau_halfcauchy_scale": tau_scale_val,
                        "sigma_y_used": sigma_y_val,
                        "sigma_c_used": sigma_c_val,
                        "mu_mean": mu_mean,
                        "mu_sd": float(mu_chains.ravel().std(ddof=1)),
                        "mu_2.5%": float(mu_ci[0]),
                        "mu_97.5%": float(mu_ci[1]),
                        "tau_mean": tau_mean,
                        "tau_sd": float(tau_chains.ravel().std(ddof=1)),
                        "tau_2.5%": float(tau_ci[0]),
                        "tau_97.5%": float(tau_ci[1]),
                        "phi_mean": phi_mean,
                        "phi_sd": float(phi_chains.ravel().std(ddof=1)),
                        "phi_2.5%": float(phi_ci[0]),
                        "phi_97.5%": float(phi_ci[1]),
                        "Z_mean": z_mean,
                        "Z_sd": float(z_mean_chains.ravel().std(ddof=1)),
                        "Z_2.5%": float(z_ci[0]),
                        "Z_97.5%": float(z_ci[1]),
                        "Rhat_mu": rhat(mu_chains),
                        "Rhat_tau": rhat(tau_chains),
                        "Rhat_phi": rhat(phi_chains),
                        "Rhat_Z_mean": rhat(z_mean_chains),
                        "tau_accept_rate": float(
                            np.mean([o["tau_accept_rate"] for o in chain_outputs])
                        ),
                        "phi_accept_rate": float(
                            np.mean([o["phi_accept_rate"] for o in chain_outputs])
                        ),
                        "z_accept_rate": float(
                            np.mean([o["z_accept_rate"] for o in chain_outputs])
                        ),
                    }

                    validation_metrics = compute_validation_metrics(
                        Y_mean,
                        Y_validation_true,
                        mask_validation,
                    )

                    row.update(
                        {f"Validation_{k}": v for k, v in validation_metrics.items()}
                    )

                    row["Validation_Coverage"] = coverage_from_chain_outputs(
                        chain_outputs,
                        Y_validation_true,
                        mask_model,
                        mask_validation,
                    )

                    if mu_true_available:
                        row["mu_bias"] = mu_mean - mu_true
                        row["mu_rmse_param"] = float(
                            np.sqrt(np.mean((mu_chains.ravel() - mu_true) ** 2))
                        )
                        row["mu_coverage"] = bool(mu_ci[0] <= mu_true <= mu_ci[1])

                    if tau_true_available:
                        row["tau_bias"] = tau_mean - tau_true
                        row["tau_rmse_param"] = float(
                            np.sqrt(np.mean((tau_chains.ravel() - tau_true) ** 2))
                        )
                        row["tau_coverage"] = bool(tau_ci[0] <= tau_true <= tau_ci[1])

                    if phi_true_available:
                        row["phi_bias"] = phi_mean - phi_true
                        row["phi_rmse_param"] = float(
                            np.sqrt(np.mean((phi_chains.ravel() - phi_true) ** 2))
                        )
                        row["phi_coverage"] = bool(phi_ci[0] <= phi_true <= phi_ci[1])

                    if Y_true_available:
                        metrics = compute_metrics(Y_mean, Y_true, mask_miss)
                        row.update({f"Bayes_{k}": v for k, v in metrics.items()})

                        y_true_miss = Y_true[mask_miss]
                        y_miss_all_truth = np.vstack(
                            [o["Y_miss_draws"] for o in chain_outputs]
                        ).astype(np.float32, copy=False)
                        lower = np.quantile(y_miss_all_truth, 0.025, axis=0)
                        upper = np.quantile(y_miss_all_truth, 0.975, axis=0)

                        row["Y_coverage_avg"] = float(
                            np.mean((y_true_miss >= lower) & (y_true_miss <= upper))
                        )

                    all_results.append(row)

                    model_objects[label] = {
                        "mu_chains": mu_chains,
                        "tau_chains": tau_chains,
                        "phi_chains": phi_chains,
                        "z_mean_chains": z_mean_chains,
                        "Y_mean": Y_mean,
                        "theta_mean": theta_mean,
                        "Z_mean_time": Z_mean_time,
                        "Z_mean_matrix": Z_mean_matrix,
                    }


results_df = pd.DataFrame(all_results)

print("\nScenario 4 completed: AR(1) structure with sampled Z")
print(results_df.to_string(index=False))


# ============================================================
# 5. Baseline comparison
# ============================================================

baseline_rows = []

Y_naive = naive_mean_impute(Y_model_input)

baseline_rows.append({
    "Method": "Naive mean",
    **compute_validation_metrics(Y_naive, Y_validation_true, mask_validation),
    "Coverage": np.nan,
})

Y_mice = mice_impute(Y_model_input, n_updates=10)

baseline_rows.append({
    "Method": "MICE" if HAVE_MICE else "MICE-style iterative regression",
    **compute_validation_metrics(Y_mice, Y_validation_true, mask_validation),
    "Coverage": np.nan,
})

Y_gnb = gaussian_nb_impute(Y_model_input, C_aligned)

baseline_rows.append({
    "Method": "Gaussian NB baseline",
    **compute_validation_metrics(Y_gnb, Y_validation_true, mask_validation),
    "Coverage": np.nan,
})

baseline_df = pd.DataFrame(baseline_rows)

print("\nBaseline method results:")
print(baseline_df.to_string(index=False))


# ============================================================
# 5a. Multi-seed validation stability check
# ============================================================

multi_seed_bayesian_rows = []
multi_seed_baseline_rows = []

for validation_seed in VALIDATION_SEEDS:
    (
        seed_mask_validation,
        seed_Y_validation_true,
        seed_Y_model_input,
        seed_mask_model,
        seed_R_model,
    ) = make_validation_split(validation_seed)

    print(f"\nRunning validation seed {validation_seed}")

    for z_prior in Z_PRIORS:
        a_z, b_z = z_prior

        for mu0_val, sigma0_val in MU_PRIORS:
            for tau_scale_val in TAU_SCALES:
                for sigma_y_val in SIGMA_Y_GRID:
                    for sigma_c_val in SIGMA_C_GRID:
                        label = (
                            f"AR1_SampledZ_Beta{int(a_z)}_{int(b_z)}_"
                            f"muSD{int(sigma0_val)}_"
                            f"HC{str(tau_scale_val).replace('.', 'p')}_"
                            f"sigY{sigma_y_val:.3g}_"
                            f"sigC{sigma_c_val:.3g}"
                        ).replace(".", "p")

                        chain_outputs = []

                        for ch in range(N_CHAINS):
                            out = run_chain_ar1_sampled_z(
                                seed_Y_model_input,
                                C_aligned,
                                seed_mask_model,
                                z_prior=z_prior,
                                mu0=mu0_val,
                                sigma0=sigma0_val,
                                tau_scale=tau_scale_val,
                                n_iter=N_ITER,
                                burn=BURN,
                                sigma_y=sigma_y_val,
                                sigma_c=sigma_c_val,
                                proposal_sd_eta_tau=0.08,
                                proposal_sd_eta_phi=0.08,
                                seed=8026
                                + validation_seed
                                + 1000 * ch
                                + int(100 * a_z)
                                + int(10 * tau_scale_val)
                                + int(round(100 * sigma_y_val))
                                + int(round(1000 * sigma_c_val)),
                            )
                            chain_outputs.append(out)

                        mu_chains = np.vstack([o["mu"] for o in chain_outputs])
                        tau_chains = np.vstack([o["tau"] for o in chain_outputs])
                        phi_chains = np.vstack([o["phi"] for o in chain_outputs])
                        z_mean_chains = np.vstack([o["Z_mean_chain"] for o in chain_outputs])
                        Y_mean = np.mean([o["Y_mean"] for o in chain_outputs], axis=0)

                        validation_metrics = compute_validation_metrics(
                            Y_mean,
                            seed_Y_validation_true,
                            seed_mask_validation,
                        )

                        multi_seed_bayesian_rows.append(
                            {
                                "Validation_Seed": validation_seed,
                                "Method_Type": "Bayesian AR(1) sampled Z",
                                "Model": label,
                                "Z_prior": f"Beta({int(a_z)},{int(b_z)})",
                                "a_z": a_z,
                                "b_z": b_z,
                                "mu_prior_sd": sigma0_val,
                                "tau_halfcauchy_scale": tau_scale_val,
                                "sigma_y_used": sigma_y_val,
                                "sigma_c_used": sigma_c_val,
                                "Z_mean": float(z_mean_chains.mean()),
                                "Rhat_mu": rhat(mu_chains),
                                "Rhat_tau": rhat(tau_chains),
                                "Rhat_phi": rhat(phi_chains),
                                "Rhat_Z_mean": rhat(z_mean_chains),
                                "tau_accept_rate": float(
                                    np.mean([o["tau_accept_rate"] for o in chain_outputs])
                                ),
                                "phi_accept_rate": float(
                                    np.mean([o["phi_accept_rate"] for o in chain_outputs])
                                ),
                                "z_accept_rate": float(
                                    np.mean([o["z_accept_rate"] for o in chain_outputs])
                                ),
                                "Validation_RMSE": validation_metrics["RMSE"],
                                "Validation_Bias": validation_metrics["Bias"],
                                "Validation_SE": validation_metrics["SE"],
                                "Validation_Coverage": coverage_from_chain_outputs(
                                    chain_outputs,
                                    seed_Y_validation_true,
                                    seed_mask_model,
                                    seed_mask_validation,
                                ),
                            }
                        )

    seed_Y_naive = naive_mean_impute(seed_Y_model_input)
    seed_Y_mice = mice_impute(seed_Y_model_input, n_updates=10)
    seed_Y_gnb = gaussian_nb_impute(seed_Y_model_input, C_aligned)

    for method_name, y_hat in [
        ("Naive mean", seed_Y_naive),
        ("MICE" if HAVE_MICE else "MICE-style iterative regression", seed_Y_mice),
        ("Gaussian NB baseline", seed_Y_gnb),
    ]:
        seed_metrics = compute_validation_metrics(
            y_hat,
            seed_Y_validation_true,
            seed_mask_validation,
        )

        multi_seed_baseline_rows.append(
            {
                "Validation_Seed": validation_seed,
                "Method_Type": "Comparison method",
                "Model": method_name,
                "Z_prior": np.nan,
                "a_z": np.nan,
                "b_z": np.nan,
                "mu_prior_sd": np.nan,
                "tau_halfcauchy_scale": np.nan,
                "sigma_y_used": np.nan,
                "sigma_c_used": np.nan,
                "Z_mean": np.nan,
                "Rhat_mu": np.nan,
                "Rhat_tau": np.nan,
                "Rhat_phi": np.nan,
                "Rhat_Z_mean": np.nan,
                "tau_accept_rate": np.nan,
                "phi_accept_rate": np.nan,
                "z_accept_rate": np.nan,
                "Validation_RMSE": seed_metrics["RMSE"],
                "Validation_Bias": seed_metrics["Bias"],
                "Validation_SE": seed_metrics["SE"],
                "Validation_Coverage": np.nan,
            }
        )

multi_seed_bayesian_df = pd.DataFrame(multi_seed_bayesian_rows)
multi_seed_baseline_df = pd.DataFrame(multi_seed_baseline_rows)
multi_seed_validation_df = pd.concat(
    [multi_seed_bayesian_df, multi_seed_baseline_df],
    ignore_index=True,
)

summary_group_cols = [
    "Method_Type",
    "Model",
    "Z_prior",
    "a_z",
    "b_z",
    "mu_prior_sd",
    "tau_halfcauchy_scale",
    "sigma_y_used",
    "sigma_c_used",
]

multi_seed_validation_summary_df = (
    multi_seed_validation_df
    .groupby(summary_group_cols, dropna=False)
    .agg(
        Validation_Splits=("Validation_Seed", "nunique"),
        Mean_RMSE=("Validation_RMSE", "mean"),
        SD_RMSE=("Validation_RMSE", "std"),
        Minimum_RMSE=("Validation_RMSE", "min"),
        Maximum_RMSE=("Validation_RMSE", "max"),
        Mean_Bias=("Validation_Bias", "mean"),
        Mean_Coverage=("Validation_Coverage", "mean"),
    )
    .reset_index()
    .sort_values("Mean_RMSE")
)

multi_seed_bayesian_df.to_csv(
    OUTDIR / "multi_seed_bayesian_validation_results.csv",
    index=False,
)

multi_seed_baseline_df.to_csv(
    OUTDIR / "multi_seed_baseline_validation_results.csv",
    index=False,
)

multi_seed_validation_summary_df.to_csv(
    OUTDIR / "multi_seed_validation_summary_all_methods.csv",
    index=False,
)

print("\nMulti-seed validation summary:")
print(multi_seed_validation_summary_df.to_string(index=False))


# ============================================================
# 6. Model comparison, diagnostics, and output files
# ============================================================

best_model_name = (
    multi_seed_validation_summary_df[
        multi_seed_validation_summary_df["Method_Type"] == "Bayesian AR(1) sampled Z"
    ]
    .sort_values("Mean_RMSE")
    .iloc[0]["Model"]
)
best_obj = model_objects[best_model_name]

print("\nBest/selected AR(1) sampled-Z model by mean multi-seed RMSE:", best_model_name)

y_miss_draw_cache = {}


def get_y_miss_all_for_model(model_name):
    if model_name in y_miss_draw_cache:
        return y_miss_draw_cache[model_name]

    model_row = results_df.loc[results_df["Model"] == model_name].iloc[0]
    chain_outputs = []

    for ch in range(N_CHAINS):
        out = run_chain_ar1_sampled_z(
            Y_model_input,
            C_aligned,
            mask_model,
            z_prior=(float(model_row["a_z"]), float(model_row["b_z"])),
            mu0=0.0,
            sigma0=float(model_row["mu_prior_sd"]),
            tau_scale=float(model_row["tau_halfcauchy_scale"]),
            n_iter=N_ITER,
            burn=BURN,
            sigma_y=float(model_row["sigma_y_used"]),
            sigma_c=float(model_row["sigma_c_used"]),
            proposal_sd_eta_tau=0.08,
            proposal_sd_eta_phi=0.08,
            seed=8026
            + 1000 * ch
            + int(100 * float(model_row["a_z"]))
            + int(10 * float(model_row["tau_halfcauchy_scale"]))
            + int(round(100 * float(model_row["sigma_y_used"])))
            + int(round(1000 * float(model_row["sigma_c_used"]))),
        )
        chain_outputs.append(out)

    y_miss_draw_cache[model_name] = np.vstack(
        [o["Y_miss_draws"] for o in chain_outputs]
    )
    return y_miss_draw_cache[model_name]


diagnostics_cols = [
    "Model", "Z_prior", "mu_prior_sd", "tau_halfcauchy_scale",
    "sigma_y_used", "sigma_c_used",
    "mu_mean", "mu_sd", "mu_2.5%", "mu_97.5%",
    "tau_mean", "tau_sd", "tau_2.5%", "tau_97.5%",
    "phi_mean", "phi_sd", "phi_2.5%", "phi_97.5%",
    "Z_mean", "Z_sd", "Z_2.5%", "Z_97.5%",
    "Rhat_mu", "Rhat_tau", "Rhat_phi", "Rhat_Z_mean",
    "tau_accept_rate", "phi_accept_rate", "z_accept_rate",
    "Validation_RMSE", "Validation_Bias", "Validation_SE", "Validation_Coverage",
]

diagnostics_df = results_df[diagnostics_cols].sort_values(
    "Validation_RMSE"
).reset_index(drop=True)

diagnostics_df.to_csv(
    DIAGNOSTICS_DIR / "all_ar1_sampledZ_model_diagnostics.csv",
    index=False,
)

bayesian_comparison_df = results_df[
    [
        "Model", "Z_prior", "mu_prior_sd", "tau_halfcauchy_scale", "Z_mean",
        "sigma_y_used", "sigma_c_used",
        "Validation_RMSE", "Validation_Bias", "Validation_SE", "Validation_Coverage",
        "Rhat_mu", "Rhat_tau", "Rhat_phi", "Rhat_Z_mean",
        "tau_accept_rate", "phi_accept_rate", "z_accept_rate",
    ]
].copy()

bayesian_comparison_df["Method_Type"] = "Bayesian AR(1) sampled Z"

baseline_comparison_df = baseline_df.rename(
    columns={
        "RMSE": "Validation_RMSE",
        "Bias": "Validation_Bias",
        "SE": "Validation_SE",
        "Coverage": "Validation_Coverage",
    }
).copy()

baseline_comparison_df["Model"] = baseline_comparison_df["Method"]
baseline_comparison_df["Z_prior"] = np.nan
baseline_comparison_df["mu_prior_sd"] = np.nan
baseline_comparison_df["tau_halfcauchy_scale"] = np.nan
baseline_comparison_df["sigma_y_used"] = np.nan
baseline_comparison_df["sigma_c_used"] = np.nan
baseline_comparison_df["Z_mean"] = np.nan
baseline_comparison_df["Rhat_mu"] = np.nan
baseline_comparison_df["Rhat_tau"] = np.nan
baseline_comparison_df["Rhat_phi"] = np.nan
baseline_comparison_df["Rhat_Z_mean"] = np.nan
baseline_comparison_df["tau_accept_rate"] = np.nan
baseline_comparison_df["phi_accept_rate"] = np.nan
baseline_comparison_df["z_accept_rate"] = np.nan
baseline_comparison_df["Method_Type"] = "Comparison method"
baseline_comparison_df = baseline_comparison_df[bayesian_comparison_df.columns]

comparison_df = pd.concat(
    [bayesian_comparison_df, baseline_comparison_df],
    ignore_index=True,
)

comparison_df = comparison_df.sort_values(
    "Validation_RMSE"
).reset_index(drop=True)

comparison_df.insert(0, "Rank_by_RMSE", np.arange(1, len(comparison_df) + 1))

comparison_df.to_csv(
    OUTDIR / "model_performance_comparison_validation.csv",
    index=False,
)

comparison_df.to_csv(
    OUTDIR / "evaluation_metrics_comparison_all_methods.csv",
    index=False,
)

best_bayesian_row = comparison_df[
    comparison_df["Method_Type"] == "Bayesian AR(1) sampled Z"
].head(1)

comparison_methods = comparison_df[
    comparison_df["Method_Type"] == "Comparison method"
]

best_vs_comparison_df = pd.concat(
    [best_bayesian_row, comparison_methods],
    ignore_index=True,
).sort_values("Validation_RMSE").reset_index(drop=True)

best_vs_comparison_df.insert(
    0,
    "Comparison_Rank_by_RMSE",
    np.arange(1, len(best_vs_comparison_df) + 1),
)

best_vs_comparison_df.to_csv(
    OUTDIR / "best_bayesian_vs_comparison_methods.csv",
    index=False,
)

completed_primary_df = direct_model_df.copy()
completed_values = Y_obs_raw.copy()
completed_values[mask_miss] = best_obj["Y_mean"][mask_miss]
completed_primary_df.loc[:, Y_COLS] = completed_values
completed_primary_df.to_csv(
    OUTDIR / "completed_primary_ar1_sampledZ_best.csv",
    index=False,
)

pd.DataFrame(
    best_obj["Z_mean_matrix"],
    columns=[f"Z_t{j+1}" for j in range(T)],
).assign(ID=subject_ids).to_csv(
    OUTDIR / "best_model_posterior_mean_Z_matrix.csv",
    index=False,
)

pd.DataFrame(
    {
        "timepoint": Y_COLS,
        "posterior_mean_Z_t": best_obj["Z_mean_time"],
    }
).to_csv(
    OUTDIR / "best_model_posterior_mean_Z_by_time.csv",
    index=False,
)

trace_df = pd.DataFrame()
for ch in range(best_obj["mu_chains"].shape[0]):
    trace_df[f"mu_chain_{ch + 1}"] = best_obj["mu_chains"][ch]
    trace_df[f"tau_chain_{ch + 1}"] = best_obj["tau_chains"][ch]
    trace_df[f"phi_chain_{ch + 1}"] = best_obj["phi_chains"][ch]
    trace_df[f"Zmean_chain_{ch + 1}"] = best_obj["z_mean_chains"][ch]

trace_df.to_csv(
    OUTDIR / "best_model_mu_tau_phi_Zmean_trace_draws.csv",
    index=False,
)

save_missing_posterior_summary(
    y_miss_all=get_y_miss_all_for_model(best_model_name),
    y_mean=best_obj["Y_mean"],
    model_mask=mask_model,
    output_path=OUTDIR / "best_model_missing_posterior_summary.csv",
    ids=subject_ids,
    y_cols=Y_COLS,
    validation_mask=mask_validation,
)

results_df.to_csv(
    OUTDIR / "scenario4_ar1_sampledZ_results.csv",
    index=False,
)

baseline_df.to_csv(
    OUTDIR / "scenario4_baseline_results.csv",
    index=False,
)

pd.DataFrame(R.astype(int), columns=Y_COLS).assign(ID=subject_ids).to_csv(
    OUTDIR / "missingness_indicator_R.csv",
    index=False,
)

pd.DataFrame(R_model.astype(int), columns=Y_COLS).assign(ID=subject_ids).to_csv(
    OUTDIR / "model_missingness_indicator_R_with_validation.csv",
    index=False,
)

pd.DataFrame(mask_validation.astype(int), columns=Y_COLS).assign(ID=subject_ids).to_csv(
    OUTDIR / "validation_holdout_mask.csv",
    index=False,
)


# ============================================================
# 7. Plots
# ============================================================

fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for ch in range(best_obj["mu_chains"].shape[0]):
    axes[0].plot(best_obj["mu_chains"][ch], alpha=0.8, label=f"chain {ch+1}")
axes[0].set_title("Trace plot: mu")
axes[0].set_xlabel("Post-burn iteration")
axes[0].set_ylabel("mu")
axes[0].legend()

for ch in range(best_obj["tau_chains"].shape[0]):
    axes[1].plot(best_obj["tau_chains"][ch], alpha=0.8, label=f"chain {ch+1}")
axes[1].set_title("Trace plot: tau")
axes[1].set_xlabel("Post-burn iteration")
axes[1].set_ylabel("tau")
axes[1].legend()

for ch in range(best_obj["phi_chains"].shape[0]):
    axes[2].plot(best_obj["phi_chains"][ch], alpha=0.8, label=f"chain {ch+1}")
axes[2].set_title("Trace plot: phi")
axes[2].set_xlabel("Post-burn iteration")
axes[2].set_ylabel("phi")
axes[2].legend()

for ch in range(best_obj["z_mean_chains"].shape[0]):
    axes[3].plot(best_obj["z_mean_chains"][ch], alpha=0.8, label=f"chain {ch+1}")
axes[3].set_title("Trace plot: mean Z")
axes[3].set_xlabel("Post-burn iteration")
axes[3].set_ylabel("mean Z")
axes[3].legend()

trace_plot_path = PLOTS_DIR / "scenario4_trace_plot_best_model.png"
save_and_maybe_show(fig, trace_plot_path)

all_trace_plot_paths = []

for model_name, obj in model_objects.items():
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for ch in range(obj["mu_chains"].shape[0]):
        axes[0].plot(obj["mu_chains"][ch], alpha=0.8, label=f"chain {ch+1}")
    axes[0].set_title(f"Trace: mu\n{model_name}")
    axes[0].set_xlabel("Post-burn iteration")
    axes[0].set_ylabel("mu")
    axes[0].legend()

    for ch in range(obj["tau_chains"].shape[0]):
        axes[1].plot(obj["tau_chains"][ch], alpha=0.8, label=f"chain {ch+1}")
    axes[1].set_title(f"Trace: tau\n{model_name}")
    axes[1].set_xlabel("Post-burn iteration")
    axes[1].set_ylabel("tau")
    axes[1].legend()

    for ch in range(obj["phi_chains"].shape[0]):
        axes[2].plot(obj["phi_chains"][ch], alpha=0.8, label=f"chain {ch+1}")
    axes[2].set_title(f"Trace: phi\n{model_name}")
    axes[2].set_xlabel("Post-burn iteration")
    axes[2].set_ylabel("phi")
    axes[2].legend()

    for ch in range(obj["z_mean_chains"].shape[0]):
        axes[3].plot(obj["z_mean_chains"][ch], alpha=0.8, label=f"chain {ch+1}")
    axes[3].set_title(f"Trace: mean Z\n{model_name}")
    axes[3].set_xlabel("Post-burn iteration")
    axes[3].set_ylabel("mean Z")
    axes[3].legend()

    trace_path = PLOTS_DIR / f"trace_plot_{safe_filename(model_name)}.png"
    save_and_maybe_show(fig, trace_path)
    all_trace_plot_paths.append(trace_path)

# RMSE bar graph
top_compare_df = comparison_df.sort_values("Validation_RMSE").head(10)

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(top_compare_df)), top_compare_df["Validation_RMSE"])
ax.set_xticks(range(len(top_compare_df)))
ax.set_xticklabels(top_compare_df["Model"], rotation=75, ha="right")
ax.set_ylabel("RMSE")
ax.set_title("AR(1) sampled-Z model-performance comparison on validation holdout")
for b, v in zip(bars, top_compare_df["Validation_RMSE"]):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
validation_rmse_plot_path = PLOTS_DIR / "validation_rmse_model_comparison.png"
save_and_maybe_show(fig, validation_rmse_plot_path)

# Bias graph
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(len(top_compare_df)), top_compare_df["Validation_Bias"])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(top_compare_df)))
ax.set_xticklabels(top_compare_df["Model"], rotation=75, ha="right")
ax.set_ylabel("Bias")
ax.set_title("Validation bias comparison")
validation_bias_plot_path = PLOTS_DIR / "validation_bias_model_comparison.png"
save_and_maybe_show(fig, validation_bias_plot_path)

# Coverage graph
coverage_df = results_df.sort_values("Validation_RMSE").head(10)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(coverage_df)), coverage_df["Validation_Coverage"])
ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
ax.set_ylim(0, 1.05)
ax.set_xticks(range(len(coverage_df)))
ax.set_xticklabels(coverage_df["Model"], rotation=75, ha="right")
ax.set_ylabel("Coverage")
ax.set_title("Bayesian 95% interval coverage on validation holdout")
coverage_plot_path = PLOTS_DIR / "validation_coverage_bayesian_models.png"
save_and_maybe_show(fig, coverage_plot_path)

# Rhat graph
rhat_df = results_df.sort_values("Validation_RMSE").head(10)
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(rhat_df))
width = 0.20
ax.bar(x - 1.5 * width, rhat_df["Rhat_mu"], width, label="mu")
ax.bar(x - 0.5 * width, rhat_df["Rhat_tau"], width, label="tau")
ax.bar(x + 0.5 * width, rhat_df["Rhat_phi"], width, label="phi")
ax.bar(x + 1.5 * width, rhat_df["Rhat_Z_mean"], width, label="mean Z")
ax.axhline(1.1, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(rhat_df["Model"], rotation=75, ha="right")
ax.set_ylabel("R-hat")
ax.set_title("Convergence diagnostics for top AR(1) sampled-Z models")
ax.legend()
rhat_plot_path = PLOTS_DIR / "rhat_diagnostics_top_models.png"
save_and_maybe_show(fig, rhat_plot_path)

# Missingness heatmap
fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(mask_miss.astype(int), aspect="auto", cmap="Greys")
ax.set_title("Original missingness pattern in direct data")
ax.set_xlabel("Timepoint")
ax.set_ylabel("Subject")
ax.set_xticks(range(T))
ax.set_xticklabels(Y_COLS)
missingness_plot_path = PLOTS_DIR / "missingness_heatmap_direct_data.png"
save_and_maybe_show(fig, missingness_plot_path)


# ============================================================
# 8. Scenario-specific sampled-Z outputs
# ============================================================

scenario_output_paths = []

for z_prior in Z_PRIORS:
    a_z, b_z = z_prior
    z_label = f"Beta_{int(a_z)}_{int(b_z)}"
    z_text = f"Beta({int(a_z)},{int(b_z)})"

    z_dir = OUTDIR / f"scenario_ar1_sampled_{z_label}"
    z_plots_dir = z_dir / "graphics"
    z_diagnostics_dir = z_dir / "diagnostics"

    z_dir.mkdir(parents=True, exist_ok=True)
    z_plots_dir.mkdir(parents=True, exist_ok=True)
    z_diagnostics_dir.mkdir(parents=True, exist_ok=True)

    z_results_df = (
        results_df[results_df["Z_prior"] == z_text]
        .sort_values("Validation_RMSE")
        .reset_index(drop=True)
    )

    z_diagnostics_df = (
        diagnostics_df[diagnostics_df["Z_prior"] == z_text]
        .sort_values("Validation_RMSE")
        .reset_index(drop=True)
    )

    z_results_path = z_dir / f"{z_label}_ar1_sampledZ_bayesian_results.csv"
    z_diagnostics_path = z_diagnostics_dir / f"{z_label}_ar1_sampledZ_model_diagnostics.csv"
    z_results_df.to_csv(z_results_path, index=False)
    z_diagnostics_df.to_csv(z_diagnostics_path, index=False)
    scenario_output_paths.extend([z_results_path, z_diagnostics_path])

    z_bayesian_comparison_df = bayesian_comparison_df[
        bayesian_comparison_df["Z_prior"] == z_text
    ].copy()

    z_comparison_df = pd.concat(
        [z_bayesian_comparison_df, baseline_comparison_df],
        ignore_index=True,
    ).sort_values("Validation_RMSE").reset_index(drop=True)
    z_comparison_df.insert(0, "Rank_by_RMSE", np.arange(1, len(z_comparison_df) + 1))

    z_comparison_path = z_dir / f"{z_label}_comparison_with_baselines.csv"
    z_comparison_df.to_csv(z_comparison_path, index=False)
    scenario_output_paths.append(z_comparison_path)

    z_best_model_name = z_results_df.loc[0, "Model"]
    z_best_obj = model_objects[z_best_model_name]

    z_completed_primary_df = direct_model_df.copy()
    z_completed_values = Y_obs_raw.copy()
    z_completed_values[mask_miss] = z_best_obj["Y_mean"][mask_miss]
    z_completed_primary_df.loc[:, Y_COLS] = z_completed_values
    z_completed_path = z_dir / f"completed_primary_ar1_sampled_{z_label}_best.csv"
    z_completed_primary_df.to_csv(z_completed_path, index=False)
    scenario_output_paths.append(z_completed_path)

    z_matrix_path = z_dir / f"{z_label}_best_posterior_mean_Z_matrix.csv"
    pd.DataFrame(
        z_best_obj["Z_mean_matrix"],
        columns=[f"Z_t{j+1}" for j in range(T)],
    ).assign(ID=subject_ids).to_csv(z_matrix_path, index=False)
    scenario_output_paths.append(z_matrix_path)

    z_time_path = z_dir / f"{z_label}_best_posterior_mean_Z_by_time.csv"
    pd.DataFrame(
        {
            "timepoint": Y_COLS,
            "posterior_mean_Z_t": z_best_obj["Z_mean_time"],
        }
    ).to_csv(z_time_path, index=False)
    scenario_output_paths.append(z_time_path)

    z_trace_df = pd.DataFrame()
    for ch in range(z_best_obj["mu_chains"].shape[0]):
        z_trace_df[f"mu_chain_{ch + 1}"] = z_best_obj["mu_chains"][ch]
        z_trace_df[f"tau_chain_{ch + 1}"] = z_best_obj["tau_chains"][ch]
        z_trace_df[f"phi_chain_{ch + 1}"] = z_best_obj["phi_chains"][ch]
        z_trace_df[f"Zmean_chain_{ch + 1}"] = z_best_obj["z_mean_chains"][ch]

    z_trace_draws_path = z_diagnostics_dir / f"{z_label}_best_mu_tau_phi_Zmean_trace_draws.csv"
    z_trace_df.to_csv(z_trace_draws_path, index=False)
    scenario_output_paths.append(z_trace_draws_path)

    z_posterior_summary_path = z_diagnostics_dir / f"{z_label}_best_missing_posterior_summary.csv"
    save_missing_posterior_summary(
        y_miss_all=get_y_miss_all_for_model(z_best_model_name),
        y_mean=z_best_obj["Y_mean"],
        model_mask=mask_model,
        output_path=z_posterior_summary_path,
        ids=subject_ids,
        y_cols=Y_COLS,
        validation_mask=mask_validation,
    )
    scenario_output_paths.append(z_posterior_summary_path)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ch in range(z_best_obj["mu_chains"].shape[0]):
        axes[0].plot(z_best_obj["mu_chains"][ch], alpha=0.8, label=f"chain {ch+1}")
    axes[0].set_title(f"Trace: mu\nBest AR(1) sampled Z {z_label}")
    axes[0].set_xlabel("Post-burn iteration")
    axes[0].set_ylabel("mu")
    axes[0].legend()

    for ch in range(z_best_obj["tau_chains"].shape[0]):
        axes[1].plot(z_best_obj["tau_chains"][ch], alpha=0.8, label=f"chain {ch+1}")
    axes[1].set_title(f"Trace: tau\nBest AR(1) sampled Z {z_label}")
    axes[1].set_xlabel("Post-burn iteration")
    axes[1].set_ylabel("tau")
    axes[1].legend()

    for ch in range(z_best_obj["phi_chains"].shape[0]):
        axes[2].plot(z_best_obj["phi_chains"][ch], alpha=0.8, label=f"chain {ch+1}")
    axes[2].set_title(f"Trace: phi\nBest AR(1) sampled Z {z_label}")
    axes[2].set_xlabel("Post-burn iteration")
    axes[2].set_ylabel("phi")
    axes[2].legend()

    for ch in range(z_best_obj["z_mean_chains"].shape[0]):
        axes[3].plot(z_best_obj["z_mean_chains"][ch], alpha=0.8, label=f"chain {ch+1}")
    axes[3].set_title(f"Trace: mean Z\nBest AR(1) sampled Z {z_label}")
    axes[3].set_xlabel("Post-burn iteration")
    axes[3].set_ylabel("mean Z")
    axes[3].legend()

    z_trace_plot_path = z_plots_dir / f"{z_label}_best_trace_plot.png"
    save_and_maybe_show(fig, z_trace_plot_path)
    scenario_output_paths.append(z_trace_plot_path)

    z_report_path = z_diagnostics_dir / f"{z_label}_scenario_report.txt"
    with open(z_report_path, "w", encoding="utf-8") as report:
        report.write(f"AR(1) structure scenario with sampled Z prior = {z_text}\n")
        report.write(f"Best model for this Z prior: {z_best_model_name}\n")
        report.write(f"Scenario directory: {z_dir}\n")
        write_table_section(report, f"Bayesian AR(1) sampled-Z results for {z_label}", z_results_df, index=False)
        write_table_section(report, f"Diagnostics for AR(1) sampled Z {z_label}", z_diagnostics_df, index=False)
        write_table_section(report, f"Comparison with baselines for AR(1) sampled Z {z_label}", z_comparison_df, index=False)
    scenario_output_paths.append(z_report_path)

    print(f"\nScenario outputs saved for AR(1) sampled Z prior = {z_text}")
    print(z_comparison_df.to_string(index=False))


# ============================================================
# 9. Overall diagnostics report and manifest
# ============================================================

diagnostics_report_path = DIAGNOSTICS_DIR / "model_diagnostics_report.txt"

with open(diagnostics_report_path, "w", encoding="utf-8") as report:
    report.write("Scenario 4: AR(1) structure with sampled Z\n")
    report.write(f"Direct data: {DIRECT_DATA}\n")
    report.write(f"Collateral data: {COLLATERAL_DATA}\n")
    report.write(f"Output directory: {OUTDIR}\n")
    report.write(f"Graphics directory: {PLOTS_DIR}\n")
    report.write(f"Diagnostics directory: {DIAGNOSTICS_DIR}\n")
    report.write(f"Primary shape: {Y_obs_raw.shape}\n")
    report.write(f"Collateral shape: {C_obs_raw.shape}\n")
    report.write(f"Missing primary entries: {int(mask_miss.sum())}\n")
    report.write(f"Validation holdout entries: {int(mask_validation.sum())}\n")
    report.write(f"sigma_y used: {SIGMA_Y}\n")
    report.write(f"sigma_c used: {SIGMA_C}\n")
    report.write(f"sigma_y tuning grid: {SIGMA_Y_GRID}\n")
    report.write(f"sigma_c tuning grid: {SIGMA_C_GRID}\n")
    report.write(f"Best Bayesian model by mean multi-seed RMSE: {best_model_name}\n")
    report.write(f"Validation seeds: {VALIDATION_SEEDS}\n")
    write_table_section(report, "Multi-seed validation summary across all methods", multi_seed_validation_summary_df, index=False)
    write_table_section(report, "Multi-seed Bayesian validation results", multi_seed_bayesian_df, index=False)
    write_table_section(report, "All AR(1) sampled-Z Bayesian model diagnostics", diagnostics_df, index=False)
    write_table_section(report, "Evaluation metrics comparison across all methods", comparison_df, index=False)
    write_table_section(report, "Best Bayesian model vs comparison methods", best_vs_comparison_df, index=False)
    write_table_section(report, "Baseline method results", baseline_df, index=False)

saved_outputs = [
    OUTDIR / "completed_primary_ar1_sampledZ_best.csv",
    OUTDIR / "scenario4_ar1_sampledZ_results.csv",
    OUTDIR / "model_performance_comparison_validation.csv",
    OUTDIR / "evaluation_metrics_comparison_all_methods.csv",
    OUTDIR / "best_bayesian_vs_comparison_methods.csv",
    OUTDIR / "best_model_missing_posterior_summary.csv",
    OUTDIR / "best_model_posterior_mean_Z_matrix.csv",
    OUTDIR / "best_model_posterior_mean_Z_by_time.csv",
    OUTDIR / "best_model_mu_tau_phi_Zmean_trace_draws.csv",
    OUTDIR / "missingness_indicator_R.csv",
    OUTDIR / "model_missingness_indicator_R_with_validation.csv",
    OUTDIR / "validation_holdout_mask.csv",
    OUTDIR / "multi_seed_bayesian_validation_results.csv",
    OUTDIR / "multi_seed_baseline_validation_results.csv",
    OUTDIR / "multi_seed_validation_summary_all_methods.csv",
    DIAGNOSTICS_DIR / "all_ar1_sampledZ_model_diagnostics.csv",
    diagnostics_report_path,
    trace_plot_path,
    missingness_plot_path,
    validation_rmse_plot_path,
    validation_bias_plot_path,
    coverage_plot_path,
    rhat_plot_path,
]

saved_outputs.extend(all_trace_plot_paths)
saved_outputs.extend(scenario_output_paths)

if not baseline_df.empty:
    saved_outputs.append(OUTDIR / "scenario4_baseline_results.csv")

output_manifest_path = OUTDIR / "saved_output_manifest.csv"

pd.DataFrame(
    {
        "Output_File": [str(path) for path in saved_outputs],
        "Exists": [Path(path).exists() for path in saved_outputs],
    }
).to_csv(output_manifest_path, index=False)

print("\nSaved files:")
for path in saved_outputs:
    print(path)

print(output_manifest_path)


# ============================================================
# 10. Compact final printed tables
# ============================================================

print("\nCompact AR(1) sampled-Z Bayesian results table:")

display_cols = [
    "Model",
    "Z_prior",
    "mu_prior_sd",
    "tau_halfcauchy_scale",
    "sigma_y_used",
    "sigma_c_used",
    "mu_mean",
    "tau_mean",
    "phi_mean",
    "Z_mean",
    "Rhat_mu",
    "Rhat_tau",
    "Rhat_phi",
    "Rhat_Z_mean",
    "tau_accept_rate",
    "phi_accept_rate",
    "z_accept_rate",
    "Validation_RMSE",
    "Validation_Bias",
    "Validation_SE",
    "Validation_Coverage",
]

print(results_df[display_cols].sort_values("Validation_RMSE").to_string(index=False))

print("\nEvaluation metrics comparison across all methods:")

comparison_display_cols = [
    "Rank_by_RMSE",
    "Method_Type",
    "Model",
    "sigma_y_used",
    "sigma_c_used",
    "Validation_RMSE",
    "Validation_Bias",
    "Validation_SE",
    "Validation_Coverage",
]

print(comparison_df[comparison_display_cols].to_string(index=False))

print("\nBest Bayesian AR(1) sampled-Z model vs comparison methods:")

best_vs_display_cols = [
    "Comparison_Rank_by_RMSE",
    "Method_Type",
    "Model",
    "sigma_y_used",
    "sigma_c_used",
    "Validation_RMSE",
    "Validation_Bias",
    "Validation_SE",
    "Validation_Coverage",
]

print(best_vs_comparison_df[best_vs_display_cols].to_string(index=False))

print("\nMulti-seed validation summary across all methods:")
multi_seed_display_cols = [
    "Method_Type",
    "Model",
    "Z_prior",
    "sigma_y_used",
    "sigma_c_used",
    "Validation_Splits",
    "Mean_RMSE",
    "SD_RMSE",
    "Minimum_RMSE",
    "Maximum_RMSE",
    "Mean_Bias",
    "Mean_Coverage",
]

print(multi_seed_validation_summary_df[multi_seed_display_cols].to_string(index=False))
