# ============================================================
# SCENARIO 2: INDEPENDENT STRUCTURE WITH SAMPLED Z
# Credibility-weighted Bayesian machine-learning imputation
#
# This script modifies the independent fixed-Z scenario to use
# sampled credibility weights under independent latent outcomes:
#
#   theta_it ~ N(mu, tau^2)
#
# Sampled Z priors considered:
#   Z_t ~ Beta(1,1), Beta(2,2), Beta(2,3), or Beta(3,2)
#
# Priors:
#   mu  ~ N(0, 5^2) or N(0, 10^2)
#   tau ~ Half-Cauchy(1), Half-Cauchy(2.5), Half-Cauchy(5)
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
    r"F:\gacheri's documents\research NCDs\BIBTEX\New folder\Simulated Datasets\Bayesian Learning Algorithm\Independent_sampledZ_results"
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

print("Display plots on screen:", DISPLAY_PLOTS)
print("Scenario: Independent structure with sampled Z")
print("Output directory:", OUTDIR)
print("Validation seeds:", VALIDATION_SEEDS)


# ============================================================
# 1. LOAD DIRECT AND COLLATERAL DATA
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
        bad_ids = model_df.loc[model_df[C_COLS].isna().all(axis=1), ID_COL].head().tolist()
        raise ValueError(
            "Some direct IDs have no matching collateral row. "
            f"Example missing IDs: {bad_ids}"
        )

    direct_model_df = direct_df.loc[direct_df[ID_COL].isin(model_df[ID_COL])].copy()
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

if mask_miss.sum() == 0:
    raise ValueError("The primary data have no missing values. Imputation evaluation is not possible.")


observed_positions = np.argwhere(~mask_miss)


# ============================================================
# 2. VALIDATION HOLDOUT FROM OBSERVED PRIMARY VALUES
# ============================================================
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

print("Primary shape:", Y_obs_raw.shape)
print("Collateral shape:", C_obs_raw.shape)
print("Original missing primary entries:", int(mask_miss.sum()))
print("Validation holdout entries:", int(mask_validation.sum()))
print("Model missing entries including holdout:", int(mask_model.sum()))
print("sigma_y used:", SIGMA_Y)
print("sigma_c used:", SIGMA_C)
print("sigma_y tuning grid:", SIGMA_Y_GRID)
print("sigma_c tuning grid:", SIGMA_C_GRID)


# ============================================================
# 3. HELPER FUNCTIONS
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


def log_target_eta_tau(eta, theta, mu_current, tau_scale):
    tau_value = np.exp(eta)
    n_total = theta.size
    ss = np.sum((theta - mu_current) ** 2)

    log_lik = -n_total * np.log(tau_value) - ss / (2.0 * tau_value**2)
    log_prior = -np.log1p((tau_value / tau_scale) ** 2)
    log_jacobian = eta

    return log_lik + log_prior + log_jacobian


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
            X_obs = np.column_stack([np.ones(obs_t.sum()), Y_imp[obs_t][:, predictors]])
            y_obs = Y_imp[obs_t, t]

            try:
                beta = np.linalg.lstsq(X_obs, y_obs, rcond=None)[0]
                X_miss = np.column_stack([np.ones(miss_t.sum()), Y_imp[miss_t][:, predictors]])
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
        df = pd.DataFrame(Y_obs, columns=[f"t{j+1}" for j in range(Y_obs.shape[1])])
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

            class_means = {k: y_obs_t[y_class == k].mean() for k in np.unique(y_class)}
            fallback = np.nanmean(y_obs_t)
            Y_imp[miss_t, t] = np.array([class_means.get(k, fallback) for k in pred_class])

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
    original_mask=None,
):
    rows = []
    positions = list(zip(*np.where(model_mask)))

    for col_idx, (i, t) in enumerate(positions):
        draws = y_miss_all[:, col_idx]
        rows.append(
            {
                "ID": ids[i],
                "timepoint": y_cols[t],
                "is_original_missing": bool(original_mask[i, t]) if original_mask is not None else False,
                "is_validation_holdout": bool(validation_mask[i, t]) if validation_mask is not None else False,
                "posterior_mean": float(y_mean[i, t]),
                "posterior_se": float(np.std(draws, ddof=1)),
                "ci_2_5": float(np.quantile(draws, 0.025)),
                "ci_97_5": float(np.quantile(draws, 0.975)),
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)


# ============================================================
# 4. SAMPLED-Z GIBBS/MH SAMPLER
# ============================================================
def run_chain_independent_sampled_z(
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
    proposal_sd_eta=0.08,
    seed=1234,
):
    rng = np.random.default_rng(seed)

    n, T = Y_obs.shape
    R_local = (~mask).astype(float)
    a_z, b_z = z_prior

    Y_curr = fill_column_means(Y_obs)
    theta = Y_curr.copy()

    mu_current = mu0

    tau_start = abs(rng.standard_cauchy()) * tau_scale
    tau_start = float(np.clip(tau_start, 0.05, 10.0))
    eta_tau = np.log(tau_start)
    tau_current = tau_start

    Z = rng.beta(a_z, b_z, size=T)

    inv_sigma_y2 = 1.0 / (sigma_y**2)
    inv_sigma_c2 = 1.0 / (sigma_c**2)
    sigma0_sq = sigma0**2

    keep = n_iter - burn
    n_miss = int(mask.sum())

    mu_draws = np.zeros(keep)
    tau_draws = np.zeros(keep)
    z_mean_draws = np.zeros(keep)
    y_miss_draws = np.zeros((keep, n_miss), dtype=np.float32)

    theta_sum = np.zeros_like(theta)
    Y_sum = np.zeros_like(Y_curr)
    Z_sum = np.zeros_like(Z)

    accept_tau = 0
    accept_z = 0
    total_z = 0
    keep_idx = 0

    for it in range(n_iter):

        # Step 8: Impute missing primary Y_it
        Y_curr[mask] = rng.normal(theta[mask], sigma_y)

        # Step 9: Update theta_it using current sampled time-specific Z_t
        tau2_current = tau_current**2

        precision = (
            (1.0 / tau2_current)
            + (Z * R_local * inv_sigma_y2)
            + ((1.0 - Z) * inv_sigma_c2)
        )

        mean_num = (
            (mu_current / tau2_current)
            + ((1.0 - Z) * C_aligned * inv_sigma_c2)
            + (Z * R_local * Y_curr * inv_sigma_y2)
        )

        var = 1.0 / precision
        theta = rng.normal(mean_num * var, np.sqrt(var))

        # Step 10: Update sampled time-specific Z_t using MH
        # Target:
        #   p(Z_t | .) is proportional to
        #   Z_t^(a-1)(1-Z_t)^(b-1) exp(Z_t * sum_i Delta_it)
        # where:
        #   Delta_it = R_it log fY(Y_it | theta_it) - log fC(C_it | theta_it)
        # Proposal:
        #   Z_new ~ Beta(a,b)
        # Since the proposal equals the prior, prior and proposal cancel in the MH ratio.
        diff_y = Y_curr - theta
        diff_c = C_aligned - theta

        log_fy = -0.5 * (diff_y**2) * inv_sigma_y2
        log_fc = -0.5 * (diff_c**2) * inv_sigma_c2

        delta_by_time = np.sum(R_local * log_fy - log_fc, axis=0)

        Z_prop = rng.beta(a_z, b_z, size=T)
        log_accept = (Z_prop - Z) * delta_by_time
        log_u = np.log(rng.random(size=T))
        accept = log_u < log_accept

        Z[accept] = Z_prop[accept]
        accept_z += int(np.sum(accept))
        total_z += Z.size

        # Step 11a: Update mu | theta, tau
        theta_vec = theta.ravel()
        post_prec = theta_vec.size / tau2_current + 1.0 / sigma0_sq
        post_mean = (
            theta_vec.sum() / tau2_current
            + mu0 / sigma0_sq
        ) / post_prec
        mu_current = rng.normal(post_mean, np.sqrt(1.0 / post_prec))

        # Step 11b: Update tau | theta, mu via MH on eta=log(tau)
        eta_prop = eta_tau + rng.normal(0.0, proposal_sd_eta)

        log_curr = log_target_eta_tau(eta_tau, theta, mu_current, tau_scale)
        log_prop = log_target_eta_tau(eta_prop, theta, mu_current, tau_scale)

        if np.log(rng.random()) < (log_prop - log_curr):
            eta_tau = eta_prop
            accept_tau += 1

        tau_current = float(np.exp(eta_tau))

        if it >= burn:
            mu_draws[keep_idx] = mu_current
            tau_draws[keep_idx] = tau_current
            z_mean_draws[keep_idx] = float(Z.mean())
            y_miss_draws[keep_idx, :] = Y_curr[mask]

            theta_sum += theta
            Y_sum += Y_curr
            Z_sum += Z
            keep_idx += 1

    return {
        "mu": mu_draws,
        "tau": tau_draws,
        "Z_mean_chain": z_mean_draws,
        "Y_miss_draws": y_miss_draws,
        "theta_mean": theta_sum / keep,
        "Y_mean": Y_sum / keep,
        "Z_mean_time": Z_sum / keep,
        "Z_mean_matrix": np.tile(Z_sum / keep, (n, 1)),
        "tau_accept_rate": accept_tau / n_iter,
        "z_accept_rate": accept_z / max(total_z, 1),
    }


# ============================================================
# 5. RUN SCENARIO 2: INDEPENDENT STRUCTURE + SAMPLED Z
# ============================================================
C_aligned = align_collateral(C_obs_raw, N, T, method=COLLATERAL_LINK)

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
                        f"Ind_SampledZ_Beta{int(a_z)}_{int(b_z)}_"
                        f"muSD{int(sigma0_val)}_"
                        f"HC{str(tau_scale_val).replace('.', 'p')}_"
                        f"sigY{sigma_y_val:.3g}_"
                        f"sigC{sigma_c_val:.3g}"
                    ).replace(".", "p")

                    chain_outputs = []

                    for ch in range(N_CHAINS):
                        out = run_chain_independent_sampled_z(
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
                            proposal_sd_eta=0.08,
                            seed=4026
                            + 1000 * ch
                            + int(100 * a_z)
                            + int(10 * tau_scale_val)
                            + int(round(100 * sigma_y_val))
                            + int(round(1000 * sigma_c_val)),
                        )
                        chain_outputs.append(out)

                    mu_chains = np.vstack([o["mu"] for o in chain_outputs])
                    tau_chains = np.vstack([o["tau"] for o in chain_outputs])
                    z_mean_chains = np.vstack([o["Z_mean_chain"] for o in chain_outputs])

                    Y_mean = np.mean([o["Y_mean"] for o in chain_outputs], axis=0)
                    theta_mean = np.mean([o["theta_mean"] for o in chain_outputs], axis=0)
                    Z_mean_matrix = np.mean([o["Z_mean_matrix"] for o in chain_outputs], axis=0)

                    mu_mean = float(mu_chains.mean())
                    tau_mean = float(tau_chains.mean())
                    z_mean = float(z_mean_chains.mean())

                    mu_ci = np.quantile(mu_chains.ravel(), [0.025, 0.975])
                    tau_ci = np.quantile(tau_chains.ravel(), [0.025, 0.975])
                    z_ci = np.quantile(z_mean_chains.ravel(), [0.025, 0.975])

                    validation_metrics = compute_validation_metrics(
                        Y_mean,
                        Y_validation_true,
                        mask_validation,
                    )

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
                        "Z_mean": z_mean,
                        "Z_sd": float(z_mean_chains.ravel().std(ddof=1)),
                        "Z_2.5%": float(z_ci[0]),
                        "Z_97.5%": float(z_ci[1]),
                        "Rhat_mu": rhat(mu_chains),
                        "Rhat_tau": rhat(tau_chains),
                        "Rhat_Z_mean": rhat(z_mean_chains),
                        "tau_accept_rate": float(np.mean([o["tau_accept_rate"] for o in chain_outputs])),
                        "z_accept_rate": float(np.mean([o["z_accept_rate"] for o in chain_outputs])),
                        "Validation_RMSE": validation_metrics["RMSE"],
                        "Validation_Bias": validation_metrics["Bias"],
                        "Validation_SE": validation_metrics["SE"],
                        "Validation_Coverage": coverage_from_chain_outputs(
                            chain_outputs,
                            Y_validation_true,
                            mask_model,
                            mask_validation,
                        ),
                    }

                    all_results.append(row)

                    model_objects[label] = {
                        "mu_chains": mu_chains,
                        "tau_chains": tau_chains,
                        "z_mean_chains": z_mean_chains,
                        "Y_mean": Y_mean,
                        "theta_mean": theta_mean,
                        "Z_mean_matrix": Z_mean_matrix,
                    }

results_sampledZ_df = pd.DataFrame(all_results)

print("\nScenario 2 completed: Independent structure with sampled Z")
print(results_sampledZ_df.to_string(index=False))


# ============================================================
# 6. COMPARISON METHODS: NAIVE MEAN, MICE, AND GAUSSIAN NAIVE BAYES
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
    "Method": "Gaussian Naive Bayes",
    **compute_validation_metrics(Y_gnb, Y_validation_true, mask_validation),
    "Coverage": np.nan,
})

baseline_sampledZ_df = pd.DataFrame(baseline_rows)

print("\nComparison method results: Naive mean, MICE, and Gaussian Naive Bayes")
print(baseline_sampledZ_df.to_string(index=False))


# ============================================================
# 6a. MULTI-SEED VALIDATION STABILITY CHECK
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
                            f"Ind_SampledZ_Beta{int(a_z)}_{int(b_z)}_"
                            f"muSD{int(sigma0_val)}_"
                            f"HC{str(tau_scale_val).replace('.', 'p')}_"
                            f"sigY{sigma_y_val:.3g}_"
                            f"sigC{sigma_c_val:.3g}"
                        ).replace(".", "p")

                        chain_outputs = []

                        for ch in range(N_CHAINS):
                            out = run_chain_independent_sampled_z(
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
                                proposal_sd_eta=0.08,
                                seed=4026
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
                                "Method_Type": "Bayesian independent sampled Z",
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
                                "Rhat_Z_mean": rhat(z_mean_chains),
                                "tau_accept_rate": float(np.mean([o["tau_accept_rate"] for o in chain_outputs])),
                                "z_accept_rate": float(np.mean([o["z_accept_rate"] for o in chain_outputs])),
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
        ("Gaussian Naive Bayes", seed_Y_gnb),
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
                "Rhat_Z_mean": np.nan,
                "tau_accept_rate": np.nan,
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
# 7. SELECT BEST MODEL AND SAVE OUTPUTS
# ============================================================
best_model_name = (
    multi_seed_validation_summary_df[
        multi_seed_validation_summary_df["Method_Type"] == "Bayesian independent sampled Z"
    ]
    .sort_values("Mean_RMSE")
    .iloc[0]["Model"]
)
best_obj = model_objects[best_model_name]

print("\nBest/selected sampled-Z model by mean multi-seed RMSE:", best_model_name)


y_miss_draw_cache = {}


def get_y_miss_all_for_model(model_name):
    if model_name in y_miss_draw_cache:
        return y_miss_draw_cache[model_name]

    model_row = results_sampledZ_df.loc[results_sampledZ_df["Model"] == model_name].iloc[0]
    chain_outputs = []

    for ch in range(N_CHAINS):
        out = run_chain_independent_sampled_z(
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
            proposal_sd_eta=0.08,
            seed=4026
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


# Diagnostics dataframe
bayesian_comparison_df = results_sampledZ_df[
    [
        "Model",
        "Z_prior",
        "a_z",
        "b_z",
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
        "Rhat_Z_mean",
        "tau_accept_rate",
        "z_accept_rate",
        "Z_mean",
    ]
].copy()
bayesian_comparison_df["Method_Type"] = "Bayesian independent sampled Z"

baseline_comparison_df = baseline_sampledZ_df.rename(
    columns={
        "RMSE": "Validation_RMSE",
        "Bias": "Validation_Bias",
        "SE": "Validation_SE",
        "Coverage": "Validation_Coverage",
    }
).copy()
baseline_comparison_df["Model"] = baseline_comparison_df["Method"]
baseline_comparison_df["Z_prior"] = np.nan
baseline_comparison_df["a_z"] = np.nan
baseline_comparison_df["b_z"] = np.nan
baseline_comparison_df["mu_prior_sd"] = np.nan
baseline_comparison_df["tau_halfcauchy_scale"] = np.nan
baseline_comparison_df["sigma_y_used"] = np.nan
baseline_comparison_df["sigma_c_used"] = np.nan
baseline_comparison_df["Rhat_mu"] = np.nan
baseline_comparison_df["Rhat_tau"] = np.nan
baseline_comparison_df["Rhat_Z_mean"] = np.nan
baseline_comparison_df["tau_accept_rate"] = np.nan
baseline_comparison_df["z_accept_rate"] = np.nan
baseline_comparison_df["Z_mean"] = np.nan
baseline_comparison_df["Method_Type"] = "Comparison method"
baseline_comparison_df = baseline_comparison_df[bayesian_comparison_df.columns]

comparison_sampledZ_df = pd.concat(
    [bayesian_comparison_df, baseline_comparison_df],
    ignore_index=True,
)
comparison_sampledZ_df = comparison_sampledZ_df.sort_values("Validation_RMSE").reset_index(drop=True)
comparison_sampledZ_df.insert(0, "Rank_by_RMSE", np.arange(1, len(comparison_sampledZ_df) + 1))

completed_primary_df = direct_model_df.copy()
completed_values = Y_obs_raw.copy()
completed_values[mask_miss] = best_obj["Y_mean"][mask_miss]
completed_primary_df.loc[:, Y_COLS] = completed_values

completed_primary_df.to_csv(
    OUTDIR / "completed_primary_independent_sampledZ_best.csv",
    index=False,
)

results_sampledZ_df.to_csv(
    OUTDIR / "scenario2_independent_sampledZ_results.csv",
    index=False,
)

baseline_sampledZ_df.to_csv(
    OUTDIR / "scenario2_sampledZ_baseline_results.csv",
    index=False,
)

comparison_sampledZ_df.to_csv(
    OUTDIR / "model_performance_comparison_validation_sampledZ.csv",
    index=False,
)

comparison_sampledZ_df.to_csv(
    OUTDIR / "evaluation_metrics_comparison_all_methods.csv",
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

# Save best model trace draws
trace_df = pd.DataFrame()
for ch in range(best_obj["mu_chains"].shape[0]):
    trace_df[f"mu_chain_{ch + 1}"] = best_obj["mu_chains"][ch]
    trace_df[f"tau_chain_{ch + 1}"] = best_obj["tau_chains"][ch]
    trace_df[f"Z_mean_chain_{ch + 1}"] = best_obj["z_mean_chains"][ch]

trace_df.to_csv(
    OUTDIR / "best_model_mu_tau_Z_trace_draws.csv",
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
    original_mask=mask_miss,
)

# Save posterior Z matrix
pd.DataFrame(best_obj["Z_mean_matrix"], columns=Y_COLS).assign(ID=subject_ids).to_csv(
    OUTDIR / "best_model_posterior_mean_Z_matrix.csv",
    index=False,
)


# ============================================================
# 8. PLOTS
# ============================================================
# Trace plot for best model
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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

for ch in range(best_obj["z_mean_chains"].shape[0]):
    axes[2].plot(best_obj["z_mean_chains"][ch], alpha=0.8, label=f"chain {ch+1}")
axes[2].set_title("Trace plot: average Z")
axes[2].set_xlabel("Post-burn iteration")
axes[2].set_ylabel("average Z")
axes[2].legend()

save_and_maybe_show(fig, PLOTS_DIR / "scenario2_trace_plot_best_model.png")

# RMSE comparison
plot_df = comparison_sampledZ_df.sort_values("Validation_RMSE").head(10)
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(plot_df)), plot_df["Validation_RMSE"])
ax.set_xticks(range(len(plot_df)))
ax.set_xticklabels(plot_df["Model"], rotation=75, ha="right")
ax.set_ylabel("RMSE on held-out observed primary values")
ax.set_title("Independent sampled-Z model vs baseline methods")

for b, v in zip(bars, plot_df["Validation_RMSE"]):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

save_and_maybe_show(fig, PLOTS_DIR / "scenario2_rmse_comparison.png")

# Coverage diagnostics
coverage_df = results_sampledZ_df.sort_values("Validation_RMSE").head(10)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(coverage_df)), coverage_df["Validation_Coverage"])
ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
ax.set_ylim(0, 1.05)
ax.set_xticks(range(len(coverage_df)))
ax.set_xticklabels(coverage_df["Model"], rotation=75, ha="right")
ax.set_ylabel("Coverage")
ax.set_title("Bayesian 95% interval coverage: sampled Z")
save_and_maybe_show(fig, PLOTS_DIR / "scenario2_validation_coverage.png")

# R-hat diagnostics
rhat_df = results_sampledZ_df.sort_values("Validation_RMSE").head(10)
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(rhat_df))
width = 0.25
ax.bar(x - width, rhat_df["Rhat_mu"], width, label="mu")
ax.bar(x, rhat_df["Rhat_tau"], width, label="tau")
ax.bar(x + width, rhat_df["Rhat_Z_mean"], width, label="mean Z")
ax.axhline(1.1, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(rhat_df["Model"], rotation=75, ha="right")
ax.set_ylabel("R-hat")
ax.set_title("Convergence diagnostics: sampled Z")
ax.legend()
save_and_maybe_show(fig, PLOTS_DIR / "scenario2_rhat_diagnostics.png")


# ============================================================
# 9. REPORTS AND MANIFEST
# ============================================================
diagnostics_report_path = DIAGNOSTICS_DIR / "scenario2_sampledZ_diagnostics_report.txt"

with open(diagnostics_report_path, "w", encoding="utf-8") as report:
    report.write("Scenario 2: Independent structure with sampled Z\n")
    report.write(f"Direct data: {DIRECT_DATA}\n")
    report.write(f"Collateral data: {COLLATERAL_DATA}\n")
    report.write(f"Output directory: {OUTDIR}\n")
    report.write(f"Primary shape: {Y_obs_raw.shape}\n")
    report.write(f"Collateral shape: {C_obs_raw.shape}\n")
    report.write(f"Original missing primary entries: {int(mask_miss.sum())}\n")
    report.write(f"Validation holdout entries: {int(mask_validation.sum())}\n")
    report.write(f"sigma_y used: {SIGMA_Y}\n")
    report.write(f"sigma_c used: {SIGMA_C}\n")
    report.write(f"sigma_y tuning grid: {SIGMA_Y_GRID}\n")
    report.write(f"sigma_c tuning grid: {SIGMA_C_GRID}\n")
    report.write(f"Validation seeds: {VALIDATION_SEEDS}\n")
    report.write(f"Best Bayesian model by mean multi-seed RMSE: {best_model_name}\n")
    write_table_section(report, "Multi-seed validation summary across all methods", multi_seed_validation_summary_df, index=False)
    write_table_section(report, "Multi-seed Bayesian validation results", multi_seed_bayesian_df, index=False)
    write_table_section(report, "All sampled-Z Bayesian model results", results_sampledZ_df, index=False)
    write_table_section(report, "Evaluation metrics comparison", comparison_sampledZ_df, index=False)
    write_table_section(report, "Baseline method results", baseline_sampledZ_df, index=False)

saved_outputs = [
    OUTDIR / "completed_primary_independent_sampledZ_best.csv",
    OUTDIR / "scenario2_independent_sampledZ_results.csv",
    OUTDIR / "scenario2_sampledZ_baseline_results.csv",
    OUTDIR / "model_performance_comparison_validation_sampledZ.csv",
    OUTDIR / "evaluation_metrics_comparison_all_methods.csv",
    OUTDIR / "multi_seed_bayesian_validation_results.csv",
    OUTDIR / "multi_seed_baseline_validation_results.csv",
    OUTDIR / "multi_seed_validation_summary_all_methods.csv",
    OUTDIR / "missingness_indicator_R.csv",
    OUTDIR / "model_missingness_indicator_R_with_validation.csv",
    OUTDIR / "validation_holdout_mask.csv",
    OUTDIR / "best_model_mu_tau_Z_trace_draws.csv",
    OUTDIR / "best_model_missing_posterior_summary.csv",
    OUTDIR / "best_model_posterior_mean_Z_matrix.csv",
    PLOTS_DIR / "scenario2_trace_plot_best_model.png",
    PLOTS_DIR / "scenario2_rmse_comparison.png",
    PLOTS_DIR / "scenario2_validation_coverage.png",
    PLOTS_DIR / "scenario2_rhat_diagnostics.png",
    diagnostics_report_path,
]

output_manifest_path = OUTDIR / "saved_output_manifest_sampledZ.csv"
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

print("\nCompact sampled-Z Bayesian results table:")
compact_cols = [
    "Model",
    "Z_prior",
    "mu_prior_sd",
    "tau_halfcauchy_scale",
    "sigma_y_used",
    "sigma_c_used",
    "mu_mean",
    "tau_mean",
    "Z_mean",
    "Rhat_mu",
    "Rhat_tau",
    "Rhat_Z_mean",
    "tau_accept_rate",
    "z_accept_rate",
    "Validation_RMSE",
    "Validation_Bias",
    "Validation_SE",
    "Validation_Coverage",
]
print(results_sampledZ_df[compact_cols].sort_values("Validation_RMSE").to_string(index=False))

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
print(comparison_sampledZ_df[comparison_display_cols].to_string(index=False))

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
