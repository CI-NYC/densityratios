# Simulate two-sample data where the true log density ratio is linear in x.
# With P0 = N(0, 1) and P1 = N(shift, 1), the Radon-Nikodym derivative is
# exp(shift * x - shift^2 / 2).
simulate_two_sample <- function(n, shift = 0.5, seed = 1) {
    set.seed(seed)
    x0 <- rnorm(n)
    x1 <- rnorm(n, mean = shift)
    x <- c(x0, x1)
    gamma0 <- c(rep_len(1, n), rep_len(0, n))
    gamma1 <- c(rep_len(0, n), rep_len(1, n))
    w <- gamma0 + gamma1
    data.frame(
        x = x,
        pseudo_outcome = gamma1 / w,
        pseudo_weight = w / 2
    )
}

fit_glm <- function(family, link = "log", n = 4000, shift = 0.5) {
    df <- simulate_two_sample(n = n, shift = shift)
    fam <- density_ratio_family(family = family, link = link)
    suppressWarnings(
        stats::glm(
            pseudo_outcome ~ x,
            weights = pseudo_weight,
            family = fam,
            data = df
        )
    )
}

test_that("KL + log link recovers the linear log density ratio", {
    shift <- 0.5
    fit <- fit_glm("kullback-leibler", link = "log", shift = shift)
    coefs <- unname(stats::coef(fit))
    expect_equal(coefs[1], -shift^2 / 2, tolerance = 0.15)
    expect_equal(coefs[2], shift, tolerance = 0.15)
    expect_true(fit$converged)
})

test_that("least-squares family converges (previously failed with sign bug)", {
    # With the original sign bug in LS dev.resids, glm.fit step-halving never
    # accepted an improving step (dev increased as the fit improved) and the
    # iteration failed to converge for non-trivial data. This check would fail
    # against the pre-fix code.
    fit <- fit_glm("least-squares", link = "log", shift = 0.3)
    expect_true(fit$converged)
    expect_true(is.finite(fit$deviance))
})

test_that("all supported families fit without error using the log link", {
    for (fam in c(
        "least-squares",
        "kullback-leibler",
        "itakura-saito",
        "negative-binomial"
    )) {
        fit <- fit_glm(fam, link = "log")
        expect_true(fit$converged, label = paste("converged:", fam))
        expect_true(is.finite(fit$deviance), label = paste("deviance:", fam))
    }
})

test_that("density.linkinv applied to the linear predictor gives the density ratio", {
    # For KL + log link, predict(type = "response") gives p = linkinv(eta), so the
    # density ratio is predict(type = "link") put through density.linkinv.
    fit <- fit_glm("kullback-leibler", link = "log", shift = 0.5)
    eta <- stats::predict(fit, type = "link")
    p <- stats::predict(fit, type = "response")
    fam <- fit$family
    # Relationship density = p / (1 - p)
    expect_equal(fam$density.linkinv(eta), p / (1 - p), tolerance = 1e-6)
})
