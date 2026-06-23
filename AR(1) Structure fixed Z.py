# UPDATED FIXED-Z VERSION: matched to AR(1) sampled-Z validation settings.
# ============================================================
# SCENARIO 3: AR(1) STRUCTURE WITH FIXED Z
# Credibility-weighted Bayesian machine-learning imputation
#
# Purpose:
#   Fit AR(1) fixed-Z credibility-weighted Bayesian imputation models and
#   compare them directly with naive mean, MICE, and Gaussian Naive Bayes
#   baselines using validation holdout RMSE.
#
# Fixed credibility weights:
#   Z = 0.9, 0.7, 0.5
#
# AR(1) latent trajectory:
#   theta_i1 ~ N(mu, tau^2)
#   theta_it | theta_i,t-1 ~ N(mu + phi(theta_i,t-1 - mu), tau^2)
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
# 0. USER SETTINGS FOR DIRECT COMPARISON WITH SAMPLED-Z SCRIPT
# ============================================================
#
# These settings are intentionally matched to AR(1) Structure Sampled Z.py
# so fixed-Z and sampled-Z can be compared directly by validation RMSE.
#
# Main difference:
#   Fixed-Z script:    uses FIXED_Z_VALUES below
#   Sampled-Z script:  samples Z_t from Beta priors
# ============================================================

DIRECT_DATA = r"F:\gacheri's documents\research NCDs\BIBTEX\New folder\Simulated Datasets\direct_dataset_simulated.csv"
COLLATERAL_DATA = r"F:\gacheri's documents\research NCDs\BIBTEX\New folder\Simulated Datasets\collateral_dataset_simulated.csv"

OUTDIR = Path(
    r"F:\gacheri's documents\research NCDs\BIBTEX\New folder\Simulated Datasets\Bayesian Learning Algorithm\AR1_fixedZ_results"
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

# Multiple validation seeds test whether the result is stable across
# different held-out observed Y values.
VALIDATION_SEEDS = [2026, 2027, 2028]
VALIDATION_SEED = 2026

COLLATERAL_LINK = "first_n"

MU_PRIORS = [(0.0, 5.0), (0.0, 10.0)]
TAU_SCALES = [1.0, 2.5, 5.0]

# Fixed credibility weights to compare against sampled-Z Beta priors.
FIXED_Z_VALUES = [0.9, 0.7, 0.5]

# MCMC settings matched to AR(1) sampled-Z screening settings.
N_CHAINS = 2
N_ITER = 1000
BURN = 300

# Optional manual overrides for working noise levels.
# Leave as None to estimate from the datasets.
SIGMA_Y_OVERRIDE = None
SIGMA_C_OVERRIDE = None

print("Scenario 3: AR(1) structure with fixed Z")
print("Display plots on screen:", DISPLAY_PLOTS)
print("Validation seeds:", VALIDATION_SEEDS)


# ============================================================
# 1. Data loading and validation setup
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
            ID_COL,
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
        "Primary Y has no missing entries. Imputation RMSE and coverage cannot be computed."
    )

observed_positions = np.argwhere(~mask_miss)


def make_validation_split(validation_seed):
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

# ============================================================
# 1a. Sigma tuning settings matched to sampled-Z script
# ============================================================
#
# These grids are defined after loading the data because they depend on
# SIGMA_Y, which is estimated from the observed direct outcomes.
SIGMA_Y_GRID = [0.8 * SIGMA_Y, SIGMA_Y, 1.2 * SIGMA_Y]
SIGMA_C_GRID = [0.5 * SIGMA_Y, 0.6 * SIGMA_Y, 0.7 * SIGMA_Y, 0.8 * SIGMA_Y]

print("Direct data:", DIRECT_DATA)
print("Collateral data:", COLLATERAL_DATA)
print("Output directory:", OUTDIR)
print("Primary shape:", Y_obs_raw.shape)
print("Collateral shape:", C_obs_raw.shape)
print("Missing primary entries:", int(mask_miss.sum()))
print("Validation holdout entries:", int(mask_validation.sum()))
print("sigma_y base:", SIGMA_Y)
print("sigma_c base:", SIGMA_C)
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
        raise ValueError("Collateral data must have the same number of timepoints as primary data.")

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


def log_target_eta_tau_ar1(eta, theta, mu_current, phi_current, tau_scale):
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
    phi_value = np.tanh(eta_phi)
    mean_trans = mu_current + phi_value * (theta[:, :-1] - mu_current)
    resid_trans = theta[:, 1:] - mean_trans
    log_lik = -np.sum(resid_trans**2) / (2.0 * tau_current**2)
    log_jacobian = np.log(max(1.0 - phi_value**2, 1e-12))
    return log_lik + log_jacobian


def sample_mu_ar1(theta, tau_current, phi_current, mu0, sigma0, rng):
    n, T = theta.shape
    tau2 = tau_current**2
    sigma0_sq = sigma0**2

    precision = 1.0 / sigma0_sq
    numerator = mu0 / sigma0_sq

    precision += n / tau2
    numerator += np.sum(theta[:, 0]) / tau2

    if T > 1:
        d = theta[:, 1:] - phi_current * theta[:, :-1]
        k = 1.0 - phi_current
        precision += (k**2) * d.size / tau2
        numerator += k * np.sum(d) / tau2

    post_var = 1.0 / precision
    post_mean = post_var * numerator
    return rng.normal(post_mean, np.sqrt(post_var))


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


def coverage_from_chain_outputs(chain_outputs, y_true, model_mask, validation_mask):
    cols = validation_draw_columns(model_mask, validation_mask)
    eval_draws = np.vstack([o["Y_miss_draws"][:, cols] for o in chain_outputs])
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
    )


def write_table_section(handle, title, df, index=False):
    handle.write(f"\n{title}\n")
    handle.write("=" * len(title) + "\n")
    if df is None or df.empty:
        handle.write("No rows available.\n")
    else:
        handle.write(df.to_string(index=index))
        handle.write("\n")


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
# 3. Baseline methods
# ============================================================

def naive_mean_impute(Y_obs):
    return fill_column_means(Y_obs)


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
            X_obs = np.column_stack([np.ones(obs_t.sum()), Y_imp[obs_t][:, predictors]])
            y_obs = Y_imp[obs_t, t]

            try:
                beta = np.linalg.lstsq(X_obs, y_obs, rcond=None)[0]
                X_miss = np.column_stack([np.ones(miss_t.sum()), Y_imp[miss_t][:, predictors]])
                Y_imp[miss_t, t] = X_miss @ beta
            except np.linalg.LinAlgError:
                Y_imp[miss_t, t] = np.nanmean(y_obs)

    return Y_imp


def mice_impute(Y_obs, n_updates=10):
    if not HAVE_MICE:
        return iterative_regression_impute(Y_obs, n_updates=n_updates)

    try:
        df = pd.DataFrame(Y_obs, columns=[f"t{j+1}" for j in range(Y_obs.shape[1])])
        imp = MICEData(df)
        for _ in range(n_updates):
            imp.update_all()
        return imp.data.to_numpy(dtype=float)
    except Exception as exc:
        print("MICE failed:", exc)
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
            q1, q2 = np.nanquantile(y_obs_t, [1 / 3, 2 / 3])
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

            class_means = {k: y_obs_t[y_class == k].mean() for k in np.unique(y_class)}
            fallback = np.nanmean(y_obs_t)
            Y_imp[miss_t, t] = np.array([class_means.get(k, fallback) for k in pred_class])
        except Exception:
            Y_imp[miss_t, t] = np.nanmean(y_obs_t)

    return Y_imp


# ============================================================
# 4. AR(1) fixed-Z sampler
# ============================================================

def update_theta_ar1_fixed_z(
    theta,
    Y_curr,
    C_aligned,
    mask,
    Z_value,
    mu_current,
    tau_current,
    phi_current,
    sigma_y,
    sigma_c,
    rng,
):
    n, T = theta.shape
    R_local = (~mask).astype(float)

    inv_tau2 = 1.0 / (tau_current**2)
    inv_sigma_y2 = 1.0 / (sigma_y**2)
    inv_sigma_c2 = 1.0 / (sigma_c**2)

    for i in range(n):
        for t in range(T):
            Ri = R_local[i, t]

            precision = Z_value * Ri * inv_sigma_y2
            precision += (1.0 - Z_value) * inv_sigma_c2

            mean_num = (1.0 - Z_value) * C_aligned[i, t] * inv_sigma_c2

            if Ri == 1.0:
                mean_num += Z_value * Y_curr[i, t] * inv_sigma_y2

            if t == 0:
                precision += inv_tau2
                mean_num += mu_current * inv_tau2

                if T > 1:
                    a_next = theta[i, t + 1] - mu_current + phi_current * mu_current
                    precision += (phi_current**2) * inv_tau2
                    mean_num += phi_current * a_next * inv_tau2

            elif t == T - 1:
                m_prev = mu_current + phi_current * (theta[i, t - 1] - mu_current)
                precision += inv_tau2
                mean_num += m_prev * inv_tau2

            else:
                m_prev = mu_current + phi_current * (theta[i, t - 1] - mu_current)
                precision += inv_tau2
                mean_num += m_prev * inv_tau2

                a_next = theta[i, t + 1] - mu_current + phi_current * mu_current
                precision += (phi_current**2) * inv_tau2
                mean_num += phi_current * a_next * inv_tau2

            var = 1.0 / precision
            theta[i, t] = rng.normal(mean_num * var, np.sqrt(var))

    return theta


def run_chain_ar1_fixed_z(
    Y_obs,
    C_aligned,
    mask,
    Z_value,
    mu0=0.0,
    sigma0=5.0,
    tau_scale=2.5,
    n_iter=1000,
    burn=300,
    sigma_y=1.0,
    sigma_c=0.6,
    proposal_sd_eta_tau=0.08,
    proposal_sd_eta_phi=0.08,
    seed=1234,
):
    rng = np.random.default_rng(seed)
    n, T = Y_obs.shape

    Y_curr = fill_column_means(Y_obs)
    theta = Y_curr.copy()

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
    y_miss_draws = np.zeros((keep, n_miss), dtype=np.float32)

    theta_sum = np.zeros_like(theta)
    Y_sum = np.zeros_like(Y_curr)

    accept_tau = 0
    accept_phi = 0
    keep_idx = 0

    for it in range(n_iter):
        Y_curr[mask] = rng.normal(theta[mask], sigma_y)

        theta = update_theta_ar1_fixed_z(
            theta=theta,
            Y_curr=Y_curr,
            C_aligned=C_aligned,
            mask=mask,
            Z_value=Z_value,
            mu_current=mu_current,
            tau_current=tau_current,
            phi_current=phi_current,
            sigma_y=sigma_y,
            sigma_c=sigma_c,
            rng=rng,
        )

        mu_current = sample_mu_ar1(
            theta=theta,
            tau_current=tau_current,
            phi_current=phi_current,
            mu0=mu0,
            sigma0=sigma0,
            rng=rng,
        )

        eta_phi_prop = eta_phi + rng.normal(0.0, proposal_sd_eta_phi)
        log_phi_curr = log_target_eta_phi_ar1(eta_phi, theta, mu_current, tau_current)
        log_phi_prop = log_target_eta_phi_ar1(eta_phi_prop, theta, mu_current, tau_current)

        if np.log(rng.random()) < (log_phi_prop - log_phi_curr):
            eta_phi = eta_phi_prop
            accept_phi += 1

        phi_current = float(np.tanh(eta_phi))
        phi_current = float(np.clip(phi_current, -0.99, 0.99))
        eta_phi = np.arctanh(phi_current)

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

        if it >= burn:
            mu_draws[keep_idx] = mu_current
            tau_draws[keep_idx] = tau_current
            phi_draws[keep_idx] = phi_current
            y_miss_draws[keep_idx, :] = Y_curr[mask]

            theta_sum += theta
            Y_sum += Y_curr
            keep_idx += 1

    return {
        "mu": mu_draws,
        "tau": tau_draws,
        "phi": phi_draws,
        "Y_miss_draws": y_miss_draws,
        "theta_mean": theta_sum / keep,
        "Y_mean": Y_sum / keep,
        "tau_accept_rate": accept_tau / n_iter,
        "phi_accept_rate": accept_phi / n_iter,
    }


# ============================================================
# 5. Model fitting helpers
# ============================================================

C_aligned = align_collateral(C_obs_raw, N, T, method=COLLATERAL_LINK)


def fit_fixed_z_grid(Y_input, model_mask, validation_true, validation_mask, validation_seed=None):
    rows = []
    objects = {}

    for Z_value in FIXED_Z_VALUES:
        for mu0_val, sigma0_val in MU_PRIORS:
            for tau_scale_val in TAU_SCALES:
                for sigma_y_val in SIGMA_Y_GRID:
                    for sigma_c_val in SIGMA_C_GRID:
                        label = (
                            f"AR1_FixedZ{Z_value}_"
                            f"muSD{int(sigma0_val)}_"
                            f"HC{str(tau_scale_val).replace('.', 'p')}_"
                            f"sigY{sigma_y_val:.3g}_"
                            f"sigC{sigma_c_val:.3g}"
                        ).replace(".", "p")

                        chain_outputs = []
                        for ch in range(N_CHAINS):
                            seed_offset = 0 if validation_seed is None else validation_seed
                            out = run_chain_ar1_fixed_z(
                                Y_input,
                                C_aligned,
                                model_mask,
                                Z_value=Z_value,
                                mu0=mu0_val,
                                sigma0=sigma0_val,
                                tau_scale=tau_scale_val,
                                n_iter=N_ITER,
                                burn=BURN,
                                sigma_y=sigma_y_val,
                                sigma_c=sigma_c_val,
                                proposal_sd_eta_tau=0.08,
                                proposal_sd_eta_phi=0.08,
                                seed=6026
                                + seed_offset
                                + 1000 * ch
                                + int(100 * Z_value)
                                + int(10 * tau_scale_val)
                                + int(round(100 * sigma_y_val))
                                + int(round(1000 * sigma_c_val)),
                            )
                            chain_outputs.append(out)

                        mu_chains = np.vstack([o["mu"] for o in chain_outputs])
                        tau_chains = np.vstack([o["tau"] for o in chain_outputs])
                        phi_chains = np.vstack([o["phi"] for o in chain_outputs])

                        Y_mean = np.mean([o["Y_mean"] for o in chain_outputs], axis=0)
                        theta_mean = np.mean([o["theta_mean"] for o in chain_outputs], axis=0)

                        mu_ci = np.quantile(mu_chains.ravel(), [0.025, 0.975])
                        tau_ci = np.quantile(tau_chains.ravel(), [0.025, 0.975])
                        phi_ci = np.quantile(phi_chains.ravel(), [0.025, 0.975])
                        validation_metrics = compute_validation_metrics(
                            Y_mean,
                            validation_true,
                            validation_mask,
                        )

                        row = {
                            "Model": label,
                            "Z": Z_value,
                            "mu_prior_sd": sigma0_val,
                            "tau_halfcauchy_scale": tau_scale_val,
                            "sigma_y_used": sigma_y_val,
                            "sigma_c_used": sigma_c_val,
                            "mu_mean": float(mu_chains.mean()),
                            "mu_sd": float(mu_chains.ravel().std(ddof=1)),
                            "mu_2.5%": float(mu_ci[0]),
                            "mu_97.5%": float(mu_ci[1]),
                            "tau_mean": float(tau_chains.mean()),
                            "tau_sd": float(tau_chains.ravel().std(ddof=1)),
                            "tau_2.5%": float(tau_ci[0]),
                            "tau_97.5%": float(tau_ci[1]),
                            "phi_mean": float(phi_chains.mean()),
                            "phi_sd": float(phi_chains.ravel().std(ddof=1)),
                            "phi_2.5%": float(phi_ci[0]),
                            "phi_97.5%": float(phi_ci[1]),
                            "Rhat_mu": rhat(mu_chains),
                            "Rhat_tau": rhat(tau_chains),
                            "Rhat_phi": rhat(phi_chains),
                            "tau_accept_rate": float(np.mean([o["tau_accept_rate"] for o in chain_outputs])),
                            "phi_accept_rate": float(np.mean([o["phi_accept_rate"] for o in chain_outputs])),
                            "Validation_RMSE": validation_metrics["RMSE"],
                            "Validation_Bias": validation_metrics["Bias"],
                            "Validation_SE": validation_metrics["SE"],
                            "Validation_Coverage": coverage_from_chain_outputs(
                                chain_outputs,
                                validation_true,
                                model_mask,
                                validation_mask,
                            ),
                        }
                        rows.append(row)

                        if validation_seed is None:
                            objects[label] = {
                                "mu_chains": mu_chains,
                                "tau_chains": tau_chains,
                                "phi_chains": phi_chains,
                                "Y_mean": Y_mean,
                                "theta_mean": theta_mean,
                            }

    return pd.DataFrame(rows), objects


def baseline_results_for_split(Y_input, validation_true, validation_mask):
    baseline_rows = []
    Y_naive = naive_mean_impute(Y_input)
    Y_mice = mice_impute(Y_input, n_updates=10)
    Y_gnb = gaussian_nb_impute(Y_input, C_aligned)

    for method_name, y_hat in [
        ("Naive mean", Y_naive),
        ("MICE" if HAVE_MICE else "MICE-style iterative regression", Y_mice),
        ("Gaussian NB baseline", Y_gnb),
    ]:
        metrics = compute_validation_metrics(y_hat, validation_true, validation_mask)
        baseline_rows.append(
            {
                "Method": method_name,
                "RMSE": metrics["RMSE"],
                "Bias": metrics["Bias"],
                "SE": metrics["SE"],
                "Coverage": np.nan,
            }
        )

    return pd.DataFrame(baseline_rows)


# ============================================================
# 6. Main model run and multi-seed validation
# ============================================================

missing_fraction = mask_miss.sum() / Y_obs_raw.size
model_missing_fraction = mask_model.sum() / Y_model_input.size
print("\nPrimary missing fraction:", round(missing_fraction, 4))
print("Model missing fraction including validation holdout:", round(model_missing_fraction, 4))
print("Fixed Z values:", FIXED_Z_VALUES)
print("MCMC settings:", {"chains": N_CHAINS, "n_iter": N_ITER, "burn": BURN})

results_df, model_objects = fit_fixed_z_grid(
    Y_model_input,
    mask_model,
    Y_validation_true,
    mask_validation,
    validation_seed=None,
)

results_df.to_csv(OUTDIR / "scenario3_ar1_fixedZ_results.csv", index=False)

baseline_df = baseline_results_for_split(
    Y_model_input,
    Y_validation_true,
    mask_validation,
)
baseline_df.to_csv(OUTDIR / "scenario3_baseline_results.csv", index=False)

print("\nSingle-seed AR(1) fixed-Z Bayesian results:")
print(results_df.sort_values("Validation_RMSE").to_string(index=False))

print("\nBaseline method results:")
print(baseline_df.to_string(index=False))

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

    seed_results_df, _ = fit_fixed_z_grid(
        seed_Y_model_input,
        seed_mask_model,
        seed_Y_validation_true,
        seed_mask_validation,
        validation_seed=validation_seed,
    )
    seed_results_df.insert(0, "Validation_Seed", validation_seed)
    seed_results_df.insert(1, "Method_Type", "Bayesian AR(1) fixed Z")
    multi_seed_bayesian_rows.append(seed_results_df)

    seed_baseline_df = baseline_results_for_split(
        seed_Y_model_input,
        seed_Y_validation_true,
        seed_mask_validation,
    )
    seed_baseline_df = seed_baseline_df.rename(
        columns={
            "Method": "Model",
            "RMSE": "Validation_RMSE",
            "Bias": "Validation_Bias",
            "SE": "Validation_SE",
            "Coverage": "Validation_Coverage",
        }
    )
    seed_baseline_df.insert(0, "Validation_Seed", validation_seed)
    seed_baseline_df.insert(1, "Method_Type", "Comparison method")
    seed_baseline_df["Z"] = np.nan
    seed_baseline_df["mu_prior_sd"] = np.nan
    seed_baseline_df["tau_halfcauchy_scale"] = np.nan
    seed_baseline_df["sigma_y_used"] = np.nan
    seed_baseline_df["sigma_c_used"] = np.nan
    multi_seed_baseline_rows.append(seed_baseline_df)

multi_seed_bayesian_df = pd.concat(multi_seed_bayesian_rows, ignore_index=True)
multi_seed_baseline_df = pd.concat(multi_seed_baseline_rows, ignore_index=True)
multi_seed_validation_df = pd.concat(
    [multi_seed_bayesian_df, multi_seed_baseline_df],
    ignore_index=True,
)

summary_group_cols = [
    "Method_Type",
    "Model",
    "Z",
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

multi_seed_bayesian_df.to_csv(OUTDIR / "multi_seed_bayesian_validation_results.csv", index=False)
multi_seed_baseline_df.to_csv(OUTDIR / "multi_seed_baseline_validation_results.csv", index=False)
multi_seed_validation_summary_df.to_csv(OUTDIR / "multi_seed_validation_summary_all_methods.csv", index=False)

print("\nMulti-seed validation summary:")
print(multi_seed_validation_summary_df.to_string(index=False))

best_model_name = (
    multi_seed_validation_summary_df[
        multi_seed_validation_summary_df["Method_Type"] == "Bayesian AR(1) fixed Z"
    ]
    .sort_values("Mean_RMSE")
    .iloc[0]["Model"]
)
best_obj = model_objects[best_model_name]

print("\nBest/selected AR(1) fixed-Z model by mean multi-seed RMSE:", best_model_name)


def get_y_miss_all_for_model(model_name):
    model_row = results_df.loc[results_df["Model"] == model_name].iloc[0]
    chain_outputs = []

    for ch in range(N_CHAINS):
        out = run_chain_ar1_fixed_z(
            Y_model_input,
            C_aligned,
            mask_model,
            Z_value=float(model_row["Z"]),
            mu0=0.0,
            sigma0=float(model_row["mu_prior_sd"]),
            tau_scale=float(model_row["tau_halfcauchy_scale"]),
            n_iter=N_ITER,
            burn=BURN,
            sigma_y=float(model_row["sigma_y_used"]),
            sigma_c=float(model_row["sigma_c_used"]),
            proposal_sd_eta_tau=0.08,
            proposal_sd_eta_phi=0.08,
            seed=6026
            + 1000 * ch
            + int(100 * float(model_row["Z"]))
            + int(10 * float(model_row["tau_halfcauchy_scale"]))
            + int(round(100 * float(model_row["sigma_y_used"])))
            + int(round(1000 * float(model_row["sigma_c_used"]))),
        )
        chain_outputs.append(out)

    return np.vstack([o["Y_miss_draws"] for o in chain_outputs])


# ============================================================
# 7. Combined comparisons and saved outputs
# ============================================================

diagnostics_cols = [
    "Model",
    "Z",
    "mu_prior_sd",
    "tau_halfcauchy_scale",
    "sigma_y_used",
    "sigma_c_used",
    "mu_mean",
    "mu_sd",
    "mu_2.5%",
    "mu_97.5%",
    "tau_mean",
    "tau_sd",
    "tau_2.5%",
    "tau_97.5%",
    "phi_mean",
    "phi_sd",
    "phi_2.5%",
    "phi_97.5%",
    "Rhat_mu",
    "Rhat_tau",
    "Rhat_phi",
    "tau_accept_rate",
    "phi_accept_rate",
    "Validation_RMSE",
    "Validation_Bias",
    "Validation_SE",
    "Validation_Coverage",
]

diagnostics_df = results_df[diagnostics_cols].sort_values("Validation_RMSE").reset_index(drop=True)
diagnostics_df.to_csv(DIAGNOSTICS_DIR / "all_ar1_fixedZ_model_diagnostics.csv", index=False)

bayesian_comparison_df = results_df[
    [
        "Model",
        "Z",
        "mu_prior_sd",
        "tau_halfcauchy_scale",
        "sigma_y_used",
        "sigma_c_used",
        "Validation_RMSE",
        "Validation_Bias",
        "Validation_SE",
        "Validation_Coverage",
        "Rhat_mu",
        "Rhat_tau",
        "Rhat_phi",
        "tau_accept_rate",
        "phi_accept_rate",
    ]
].copy()
bayesian_comparison_df["Method_Type"] = "Bayesian AR(1) fixed Z"

baseline_comparison_df = baseline_df.rename(
    columns={
        "Method": "Model",
        "RMSE": "Validation_RMSE",
        "Bias": "Validation_Bias",
        "SE": "Validation_SE",
        "Coverage": "Validation_Coverage",
    }
).copy()

baseline_comparison_df["Z"] = np.nan
baseline_comparison_df["mu_prior_sd"] = np.nan
baseline_comparison_df["tau_halfcauchy_scale"] = np.nan
baseline_comparison_df["sigma_y_used"] = np.nan
baseline_comparison_df["sigma_c_used"] = np.nan
baseline_comparison_df["Rhat_mu"] = np.nan
baseline_comparison_df["Rhat_tau"] = np.nan
baseline_comparison_df["Rhat_phi"] = np.nan
baseline_comparison_df["tau_accept_rate"] = np.nan
baseline_comparison_df["phi_accept_rate"] = np.nan
baseline_comparison_df["Method_Type"] = "Comparison method"
baseline_comparison_df = baseline_comparison_df[bayesian_comparison_df.columns]

comparison_df = pd.concat(
    [bayesian_comparison_df, baseline_comparison_df],
    ignore_index=True,
).sort_values("Validation_RMSE").reset_index(drop=True)
comparison_df.insert(0, "Rank_by_RMSE", np.arange(1, len(comparison_df) + 1))

comparison_df.to_csv(OUTDIR / "model_performance_comparison_validation.csv", index=False)
comparison_df.to_csv(OUTDIR / "evaluation_metrics_comparison_all_methods.csv", index=False)

best_bayesian_row = comparison_df[
    comparison_df["Method_Type"] == "Bayesian AR(1) fixed Z"
].head(1)
comparison_methods = comparison_df[comparison_df["Method_Type"] == "Comparison method"]
best_vs_comparison_df = pd.concat(
    [best_bayesian_row, comparison_methods],
    ignore_index=True,
).sort_values("Validation_RMSE").reset_index(drop=True)
best_vs_comparison_df.insert(
    0,
    "Comparison_Rank_by_RMSE",
    np.arange(1, len(best_vs_comparison_df) + 1),
)
best_vs_comparison_df.to_csv(OUTDIR / "best_bayesian_vs_comparison_methods.csv", index=False)

completed_primary_df = direct_model_df.copy()
completed_values = Y_obs_raw.copy()
completed_values[mask_miss] = best_obj["Y_mean"][mask_miss]
completed_primary_df.loc[:, Y_COLS] = completed_values
completed_primary_df.to_csv(OUTDIR / "completed_primary_ar1_fixedZ_best.csv", index=False)

trace_df = pd.DataFrame()
for ch in range(best_obj["mu_chains"].shape[0]):
    trace_df[f"mu_chain_{ch + 1}"] = best_obj["mu_chains"][ch]
    trace_df[f"tau_chain_{ch + 1}"] = best_obj["tau_chains"][ch]
    trace_df[f"phi_chain_{ch + 1}"] = best_obj["phi_chains"][ch]
trace_df.to_csv(OUTDIR / "best_model_mu_tau_phi_trace_draws.csv", index=False)

save_missing_posterior_summary(
    y_miss_all=get_y_miss_all_for_model(best_model_name),
    y_mean=best_obj["Y_mean"],
    model_mask=mask_model,
    output_path=OUTDIR / "best_model_missing_posterior_summary.csv",
    ids=subject_ids,
    y_cols=Y_COLS,
    validation_mask=mask_validation,
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
# 8. Graphics and report
# ============================================================

top_compare_df = comparison_df.sort_values("Validation_RMSE").head(10)

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(top_compare_df)), top_compare_df["Validation_RMSE"])
ax.set_xticks(range(len(top_compare_df)))
ax.set_xticklabels(top_compare_df["Model"], rotation=75, ha="right")
ax.set_ylabel("RMSE")
ax.set_title("AR(1) fixed-Z model-performance comparison on validation holdout")
for b, v in zip(bars, top_compare_df["Validation_RMSE"]):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
validation_rmse_plot_path = PLOTS_DIR / "validation_rmse_model_comparison.png"
save_and_maybe_show(fig, validation_rmse_plot_path)

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

rhat_df = results_df.sort_values("Validation_RMSE").head(10)
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(rhat_df))
width = 0.25
ax.bar(x - width, rhat_df["Rhat_mu"], width, label="mu")
ax.bar(x, rhat_df["Rhat_tau"], width, label="tau")
ax.bar(x + width, rhat_df["Rhat_phi"], width, label="phi")
ax.axhline(1.1, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(rhat_df["Model"], rotation=75, ha="right")
ax.set_ylabel("R-hat")
ax.set_title("Convergence diagnostics for top AR(1) fixed-Z models")
ax.legend()
rhat_plot_path = PLOTS_DIR / "rhat_diagnostics_top_models.png"
save_and_maybe_show(fig, rhat_plot_path)

fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(mask_miss.astype(int), aspect="auto", cmap="Greys")
ax.set_title("Original missingness pattern in direct data")
ax.set_xlabel("Timepoint")
ax.set_ylabel("Subject")
ax.set_xticks(range(T))
ax.set_xticklabels(Y_COLS)
missingness_plot_path = PLOTS_DIR / "missingness_heatmap_direct_data.png"
save_and_maybe_show(fig, missingness_plot_path)

diagnostics_report_path = DIAGNOSTICS_DIR / "model_diagnostics_report.txt"
with open(diagnostics_report_path, "w", encoding="utf-8") as report:
    report.write("Scenario 3: AR(1) structure with fixed Z\n")
    report.write(f"Direct data: {DIRECT_DATA}\n")
    report.write(f"Collateral data: {COLLATERAL_DATA}\n")
    report.write(f"Output directory: {OUTDIR}\n")
    report.write(f"Graphics directory: {PLOTS_DIR}\n")
    report.write(f"Diagnostics directory: {DIAGNOSTICS_DIR}\n")
    report.write(f"Primary shape: {Y_obs_raw.shape}\n")
    report.write(f"Collateral shape: {C_obs_raw.shape}\n")
    report.write(f"Missing primary entries: {int(mask_miss.sum())}\n")
    report.write(f"Validation holdout entries: {int(mask_validation.sum())}\n")
    report.write(f"sigma_y tuning grid: {SIGMA_Y_GRID}\n")
    report.write(f"sigma_c tuning grid: {SIGMA_C_GRID}\n")
    report.write(f"Validation seeds: {VALIDATION_SEEDS}\n")
    report.write(f"Best Bayesian model by mean multi-seed RMSE: {best_model_name}\n")
    write_table_section(report, "Multi-seed validation summary across all methods", multi_seed_validation_summary_df, index=False)
    write_table_section(report, "Single-seed Bayesian model diagnostics", diagnostics_df, index=False)
    write_table_section(report, "Evaluation metrics comparison across all methods", comparison_df, index=False)
    write_table_section(report, "Best Bayesian model vs comparison methods", best_vs_comparison_df, index=False)
    write_table_section(report, "Baseline method results", baseline_df, index=False)

saved_outputs = [
    OUTDIR / "completed_primary_ar1_fixedZ_best.csv",
    OUTDIR / "scenario3_ar1_fixedZ_results.csv",
    OUTDIR / "scenario3_baseline_results.csv",
    OUTDIR / "model_performance_comparison_validation.csv",
    OUTDIR / "evaluation_metrics_comparison_all_methods.csv",
    OUTDIR / "best_bayesian_vs_comparison_methods.csv",
    OUTDIR / "best_model_missing_posterior_summary.csv",
    OUTDIR / "best_model_mu_tau_phi_trace_draws.csv",
    OUTDIR / "missingness_indicator_R.csv",
    OUTDIR / "model_missingness_indicator_R_with_validation.csv",
    OUTDIR / "validation_holdout_mask.csv",
    OUTDIR / "multi_seed_bayesian_validation_results.csv",
    OUTDIR / "multi_seed_baseline_validation_results.csv",
    OUTDIR / "multi_seed_validation_summary_all_methods.csv",
    DIAGNOSTICS_DIR / "all_ar1_fixedZ_model_diagnostics.csv",
    diagnostics_report_path,
    validation_rmse_plot_path,
    coverage_plot_path,
    rhat_plot_path,
    missingness_plot_path,
]

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

print("\nCompact AR(1) fixed-Z Bayesian results table:")
display_cols = [
    "Model",
    "Z",
    "mu_prior_sd",
    "tau_halfcauchy_scale",
    "sigma_y_used",
    "sigma_c_used",
    "mu_mean",
    "tau_mean",
    "phi_mean",
    "Rhat_mu",
    "Rhat_tau",
    "Rhat_phi",
    "tau_accept_rate",
    "phi_accept_rate",
    "Validation_RMSE",
    "Validation_Bias",
    "Validation_SE",
    "Validation_Coverage",
]
print(results_df[display_cols].sort_values("Validation_RMSE").to_string(index=False))

print("\nMulti-seed validation summary across all methods:")
multi_seed_display_cols = [
    "Method_Type",
    "Model",
    "Z",
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
