import timeit
from copy import deepcopy
from pathlib import Path

import altair as alt
import click
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import yaml

from density_ratios.augmentation import (
    augment_shift_intervention,
    augment_stabilized_weights,
)
from density_ratios.logging import get_logger
from density_ratios.objectives import BinaryCrossEntropy, KullbackLeibler, LeastSquares
from density_ratios.train import train

logger = get_logger(__name__)

MODEL_DIR = Path("paper/models")
RESULTS_DIR = Path("results")


def dgp1_treatment_mean(x, c=0.5):
    # E[A|X]
    return c * x[..., 0]


def dgp1(key, num_samples, num_features=20):
    x = jax.random.normal(key, shape=(num_samples, num_features))
    a = jax.random.normal(key, shape=(num_samples,)) + dgp1_treatment_mean(x)
    return a.reshape((-1, 1)), x


def dgp1_true_ratio_shift(a, x, shift_size: float = 0.1):
    """True ratio function for shift intervention."""
    # p(a - shift | x) / p(a | x)
    a_residual = a.squeeze() - dgp1_treatment_mean(x)
    log_a_cond = jax.scipy.stats.norm.logpdf(a_residual)
    log_a_cond_shifted = jax.scipy.stats.norm.logpdf(a_residual - shift_size)
    return jnp.exp(log_a_cond_shifted - log_a_cond)


def dgp1_true_ratio_stabilized_weight(a, x):
    """True ratio function."""
    # p(a) / p(a | x)
    c = 0.5
    a_residual = a.squeeze() - dgp1_treatment_mean(x, c)
    log_a_cond = jax.scipy.stats.norm.logpdf(a_residual)
    marginal_var = 1 + jnp.square(c)
    log_a_marginal = jax.scipy.stats.norm.logpdf(
        a.squeeze() / jnp.sqrt(marginal_var)
    ) - (jnp.log(marginal_var) / 2)
    return jnp.exp(log_a_marginal - log_a_cond)


def dgp1_true_outcome(a, x, key):
    mu = a.squeeze() * (1 + x[..., 0]) + x[..., 0] * x[..., 1] + x[..., 2]
    num_samples = len(mu)
    return jax.random.normal(key, shape=(num_samples,)) + mu


# Function look up dict
_augmentation_funcs = {
    "augment_shift_intervention": augment_shift_intervention,
    "augment_stabilized_weights": augment_stabilized_weights,
}

_dgp_lookup = {
    "dgp1_shift": {
        "dgp": dgp1,
        "true_ratio": dgp1_true_ratio_shift,
        "true_outcome": dgp1_true_outcome,
    },
    "dgp1_stabilized_weight": {
        "dgp": dgp1,
        "true_ratio": dgp1_true_ratio_stabilized_weight,
        "true_outcome": dgp1_true_outcome,
    },
}


def augment_and_fit(
    a_train,
    x_train,
    a_valid,
    x_valid,
    a_test,
    x_test,
    augmentation_params: dict,
    param_set: dict[str, dict] | None = None,
    verbose: bool = False,
):
    param_set = param_set or {}

    augmentation_func_name = augmentation_params.get("function", "")
    augmentation_func_kwargs = augmentation_params.get("kwargs", {})
    augmentation_func = _augmentation_funcs.get(augmentation_func_name, None)
    if augmentation_func is None:
        raise KeyError(f"Augmentation function: '{augmentation_func_name}' not found.")

    # train and validation sets for algorithms that use early stopping
    delta_train, x_augmented_train, w_augmented_train = augmentation_func(
        x_train,
        a_train,
        **augmentation_func_kwargs,
    )
    delta_valid, x_augmented_valid, w_augmented_valid = augmentation_func(
        x_valid,
        a_valid,
        **augmentation_func_kwargs,
    )

    # combined train and validation set for algorithms that do not use early stopping
    delta, x_augmented, w_augmented = augmentation_func(
        np.vstack([x_train, x_valid]),
        np.vstack([a_train, a_valid]),
        **augmentation_func_kwargs,
    )

    test_data = np.column_stack([a_test, x_test])

    out = {"ones": (np.zeros(len(test_data)), 0.0)}  # log(1) = 0 as baseline
    objectives = {
        "least_squares": LeastSquares(),
        "kullback_leibler": KullbackLeibler(),
        "cross_entropy": BinaryCrossEntropy(),
    }

    for name, params in param_set.items():
        jax.clear_caches()
        model_name = params.get("model")
        objective = params.get("objective", None)

        if isinstance(objective, str):
            objective = [objective]

        if isinstance(objective, list):
            for obj_name in objective:
                obj = objectives.get(obj_name)
                if obj is None:
                    raise ValueError(f"Objective {obj_name} not found.")

                # These ones use train and validation
                # TODO make this more robust
                t0 = timeit.default_timer()
                model = train(
                    y=delta_train,
                    x=x_augmented_train,
                    weights=w_augmented_train,
                    params=params,
                    model=model_name,
                    objective=obj,
                    y_valid=delta_valid,
                    x_valid=x_augmented_valid,
                    weights_valid=w_augmented_valid,
                    verbose=verbose,
                )
                duration = timeit.default_timer() - t0
                out[f"{name}_{obj_name}"] = model.predict(test_data), duration

        else:
            t0 = timeit.default_timer()
            model = train(
                y=delta,
                x=x_augmented,
                weights=w_augmented,
                params=params,
                model=model_name,
                verbose=verbose,
            )
            duration = timeit.default_timer() - t0
            out[f"{name}"] = model.predict(test_data), duration

    return out


def evaluations(true_logits, pred_logits, true_y=None) -> dict:
    diff_logit = pred_logits - true_logits
    diff_ratio = np.exp(pred_logits) - np.exp(true_logits)
    out = {
        "rmse_logit": np.sqrt(np.mean(np.square(diff_logit))),
        "rmse_ratio": np.sqrt(np.mean(np.square(diff_ratio))),
        "mae_logit": np.mean(np.abs(diff_logit)),
        "mae_ratio": np.mean(np.abs(diff_ratio)),
    }
    if true_y is None:
        return out

    out["bias_ipw"] = np.mean(true_y * diff_ratio)
    out["mae_ipw"] = np.abs(out["bias_ipw"])
    return out


def evaluate_mulitple_predictions(
    true_logits, pred_dict: dict[str, tuple], true_y=None
) -> dict:
    stats = {
        f"{name}_{stat_name}": stat
        for name, (pred, duration) in pred_dict.items()
        for stat_name, stat in evaluations(true_logits, pred, true_y).items()
    }
    durations = {
        f"{name}_duration_seconds": duration
        for name, (pred, duration) in pred_dict.items()
    }
    stats.update(durations)
    return stats


def run_simulations(
    models_dict: dict,
    key,
    num_simulations: int,
    num_samples: int,
    num_test_samples: int = 10_000,
) -> pl.DataFrame:
    verbose = num_simulations == 1
    logger.info(
        f"Running simulations with:\n{num_simulations=}\n{num_samples=}\n{num_test_samples=}\n{verbose=}"
    )

    augmentation_params = models_dict.pop("augmentation_params", {})

    dgp_name = models_dict.pop("data_generating_process", "")
    _dgp = _dgp_lookup.get(dgp_name, None)

    if _dgp is None:
        raise KeyError(f"Data generating process: '{dgp_name}' not found.")

    dgp = _dgp.get("dgp")
    true_ratio = _dgp.get("true_ratio")
    true_outcome = _dgp.get("true_outcome")

    param_set = models_dict
    outputs = []

    for iteration, k in enumerate(jax.random.split(key, num_simulations)):
        logger.info(f"Starting simulation: {iteration}")

        # Issue when running on EC2 instance after around 20 iterations
        # See also https://github.com/jax-ml/jax/issues/11923
        jax.clear_caches()

        k_train, k_valid, k_test, k_outcome = jax.random.split(k, 4)
        # Use 80/20 test validation split
        num_train = int(num_samples * 0.8)
        a_train, x_train = dgp(k_train, num_samples=num_train)
        a_valid, x_valid = dgp(k_valid, num_samples=num_samples - num_train)
        a_test, x_test = dgp(k_test, num_samples=num_test_samples)

        true_ratios = true_ratio(a_test, x_test)
        true_y = true_outcome(a_test, x_test, k_outcome)
        pred_ratios_logit = augment_and_fit(
            a_train,
            x_train,
            a_valid,
            x_valid,
            a_test,
            x_test,
            augmentation_params,
            param_set=param_set,
            verbose=verbose,
        )
        stats = evaluate_mulitple_predictions(
            np.log(true_ratios), pred_ratios_logit, true_y
        )
        outputs.append(stats)

        # little hack for test plotting
        # todo: remove?
        if num_simulations == 1:
            return true_ratios, pred_ratios_logit, stats

    return pl.from_records(outputs)


@click.group()
def cli():
    """Main CLI group"""
    pass


@cli.command()
@click.option("--model-file", default="shift.yaml", help="Name of the model file.")
@click.option("--num-simulations", default=100, help="Number of dataset replicates.")
@click.option("--num-samples", default=2_000, help="Number of training observations.")
@click.option("--num-test-samples", default=10_000, help="Number of test observations.")
@click.option("--seed", default=123, help="Random seed.")
def simulations(model_file, num_simulations, num_samples, num_test_samples, seed: int):
    experiment_name = model_file.removesuffix(".yaml")
    experiment_name = f"{experiment_name}_{num_samples}"
    key = jax.random.PRNGKey(seed)

    with open(MODEL_DIR / model_file) as f:
        models_dict = yaml.safe_load(f)

    models_dicts = {experiment_name: models_dict}

    # Iterate over multipler_monte_carlo if necessary
    augmentation_params = models_dict.get("augmentation_params", {})
    is_sw = augmentation_params.get("function", "") == "augment_stabilized_weights"
    if is_sw:
        multipliers = augmentation_params.get("kwargs", {}).get("multipler_monte_carlo")
        if isinstance(multipliers, list):
            # create a copy of the models_dict for each multiplier

            def _modified_model_dict(multiplier):
                md = deepcopy(models_dict)
                md["augmentation_params"]["kwargs"]["multipler_monte_carlo"] = (
                    multiplier
                )
                return md

            models_dicts = {
                f"{experiment_name}_mult{m}": _modified_model_dict(m)
                for m in multipliers
            }

    for exp_name, md in models_dicts.items():
        logger.info(f"Running experiment: {exp_name}")
        t0 = timeit.default_timer()
        results = run_simulations(
            md,
            key=key,
            num_simulations=num_simulations,
            num_samples=num_samples,
            num_test_samples=num_test_samples,
        )
        duration = timeit.default_timer() - t0
        duration_mins = duration / 60
        logger.info(f"Finished experiment with duration: {duration_mins:.0f} minutes.")
        results_file = RESULTS_DIR / f"experiment_{exp_name}.parquet"
        results.write_parquet(results_file)
        logger.info(f"Simulation results written to: {str(results_file.resolve())}")

        name_as_list = pl.col("variable").str.split("_")
        res = (
            results.mean()
            .unpivot()
            .select(
                name=pl.col("variable").str.extract(
                    r"^(\w+)_(mse|rmse|mae|bias|duration)_(logit|ratio|ipw|seconds)$"
                ),
                scale=name_as_list.list.get(-1),
                stat=name_as_list.list.get(-2),
                value=pl.col("value"),
            )
            .with_columns(value_3sf=pl.col("value").round_sig_figs(3))
            .sort(["scale", "stat", "value"])
        )
        res_file = RESULTS_DIR / f"results_{exp_name}.parquet"
        res.write_parquet(res_file)
        logger.info(f"Simulation results summary written to: {str(res_file.resolve())}")


def bin_and_mean(x, y, n_bins, bin_type: str = "quantile"):
    """Get mean of y for each x bin."""
    if bin_type == "quantile":
        probs = np.linspace(0, 1, num=n_bins + 1)
        bins = np.quantile(x, probs)

    if bin_type == "linear":
        bins = np.linspace(min(x), max(x), num=n_bins + 1)

    if bin_type == "linear_y":
        bins = np.linspace(min(y), max(y), num=n_bins + 1)

    midpoints = (bins[1:] + bins[:-1]) / 2
    bin_labels = [str(i) for i in range(n_bins)]

    try:
        res = (
            pl.DataFrame({"x": np.asarray(x), "y": np.asarray(y)})
            .lazy()
            .group_by(pl.col("x").cut(bins[1:-1], labels=bin_labels))
            .agg(mean=pl.mean("y"), n=pl.len())
            # join on labels to account for empty bins
            .join(
                pl.DataFrame([pl.Series("x", bin_labels, dtype=pl.Categorical)]).lazy(),
                on="x",
                how="right",
            )
            .sort(pl.col("x").cast(pl.Int16))
            .collect()
        )
    except pl.exceptions.DuplicateError:
        return midpoints, np.zeros_like(midpoints), np.zeros_like(midpoints)

    y_means = res.select("mean").to_series().to_numpy()
    n = res.select("n").to_series().to_numpy()
    return midpoints, y_means, n


@cli.command()
@click.option("--model-file", default="shift.yaml", help="Name of the model file.")
@click.option("--num-samples", default=2_000, help="Number of training observations.")
@click.option("--seed", default=123, help="Random seed.")
def plot(model_file, num_samples, seed: int):
    experiment_name = model_file.removesuffix(".yaml")
    experiment_name = f"{experiment_name}_{num_samples}"
    key = jax.random.PRNGKey(seed)

    with open(MODEL_DIR / model_file) as f:
        models_dict = yaml.safe_load(f)

    augmentation_params = models_dict.get("augmentation_params", {})
    is_sw = augmentation_params.get("function", "") == "augment_stabilized_weights"
    if is_sw:
        multipliers = augmentation_params.get("kwargs", {}).get("multipler_monte_carlo")
        if isinstance(multipliers, list):
            # create plots only for the first multiplier value
            multiplier = multipliers[0]
            models_dict["augmentation_params"]["kwargs"]["multipler_monte_carlo"] = (
                multiplier
            )

    truth, preds, stats = run_simulations(
        models_dict,
        key,
        num_samples=num_samples,
        num_simulations=1,
    )
    # log some comparison stats
    msgs = [f"Evaluation statistics for: {experiment_name}"]
    msgs += [f"{name}: {val}" for name, val in stats.items()]
    logger.info("\n".join(msgs))

    n_bins = 15
    bin_type = "linear_y"
    bin_preds = {
        name: bin_and_mean(np.exp(pred), truth, n_bins, bin_type)
        for name, (pred, _) in preds.items()
    }

    df = (
        pl.DataFrame(
            {name: np.column_stack(x)[..., :3] for name, x in bin_preds.items()}
        )
        .unpivot()
        .select(
            method=pl.col("variable"),
            midpoint=pl.col("value").arr.get(0),
            ave_truth=pl.col("value").arr.get(1),
            n=pl.col("value").arr.get(2).fill_nan(None).cast(pl.Int32),
        )
        .filter(pl.col("method") != "ones")
    )

    chart = (
        alt.Chart(df, width=600, height=400)
        .mark_point()
        .encode(
            x=alt.X("midpoint:Q", title="Prediction (binned)"),
            y=alt.Y(
                "ave_truth:Q",
                title="True Density Ratio (mean in bin)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("method:N", title="Method"),
            size=alt.Size("n:Q").scale(rangeMax=200),
        )
        .interactive()
    )
    lines = (
        alt.Chart(df, width=600, height=400)
        .mark_line()
        .encode(
            x=alt.X("midpoint:Q", title="Prediction (binned)"),
            y=alt.Y(
                "ave_truth:Q",
                title="True Density Ratio (mean in bin)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("method:N", title="Method"),
        )
    )
    chart += lines
    chart += (
        alt.Chart(df)
        .mark_line(strokeDash=[5, 5], color="black")
        .encode(x="midpoint:Q", y="midpoint:Q")
    )

    save_path = RESULTS_DIR / f"{experiment_name}.html"
    logger.info(f"Saving plot to {str(save_path.resolve())}")
    chart.save(save_path)


if __name__ == "__main__":
    cli()
