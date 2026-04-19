# Bregman generating function second derivatives on the density-ratio scale.
# V(p) on the probability scale must equal (1 - p)^3 / F''(p / (1 - p)).
F_doubleprime <- list(
    `least-squares` = function(t) rep_len(1, length(t)),
    `kullback-leibler` = function(t) 1 / t,
    `itakura-saito` = function(t) 1 / t^2,
    `negative-binomial` = function(t) 1 / (t * (1 + t))
)

test_that("variance functions satisfy V(p) = (1-p)^3 / F''(p/(1-p))", {
    p <- seq(0.05, 0.95, length.out = 9)
    for (family in names(F_doubleprime)) {
        fam <- density_ratio_family(family)
        expected <- (1 - p)^3 / F_doubleprime[[family]](p / (1 - p))
        expect_equal(
            fam$variance(p), expected,
            tolerance = 1e-10,
            label = paste("variance", family)
        )
    }
})

test_that("dev.resids is minimised at mu == y", {
    # dev.resids equals the Bregman divergence up to an additive function of y
    # (for KL and IS). As a function of mu, it must attain its minimum at y.
    for (family in names(F_doubleprime)) {
        fam <- density_ratio_family(family)
        y <- 0.35
        mu_vals <- seq(0.05, 0.95, by = 0.025)
        devs <- vapply(
            mu_vals,
            function(m) fam$dev.resids(y = y, mu = m, wt = 1),
            numeric(1)
        )
        expect_equal(
            mu_vals[which.min(devs)], y,
            tolerance = 0.025,
            label = paste("min at y:", family)
        )
    }
})

test_that("LS dev.resids matches the Bregman form up to a y-only constant", {
    # With F_tilde(t) = t^2 / (2 (1 - t)) we have
    #   2 * (-F_tilde(mu) - F_tilde'(mu) * (y - mu)) = (mu^2 (1 + y) - 2 mu y) / (1 - mu)^2
    fam <- density_ratio_family("least-squares")
    y <- c(0.0, 0.2, 0.5, 0.75, 1.0)
    mu <- c(0.4, 0.4, 0.5, 0.5, 0.5)
    wt <- c(1, 1, 2, 0.5, 1)
    expected <- wt * (mu^2 * (1 + y) - 2 * mu * y) / (1 - mu)^2
    expect_equal(fam$dev.resids(y, mu, wt), expected, tolerance = 1e-10)

    # Finite at y = 1 (unlike the full Bregman, which is +Inf there).
    expect_true(all(is.finite(fam$dev.resids(y, mu, wt))))
})

test_that("LS dev.resids has the opposite sign of the pre-0.1.0 formula", {
    # Regression test: the prior code had mu * (2 * (y - mu) + mu * (1 - y)) / (1 - mu)^2 / 2
    # times 2, which equals the negative of the current (correct) formula.
    fam <- density_ratio_family("least-squares")
    y <- c(0.1, 0.4, 0.8)
    mu <- c(0.3, 0.5, 0.6)
    old_formula <- mu * (2 * (y - mu) + mu * (1 - y)) / (1 - mu)^2
    expect_equal(fam$dev.resids(y, mu, 1), -old_formula, tolerance = 1e-10)
})
