# Helper: check link inverse round-trips on a plausible density-ratio grid.
check_link_roundtrip <- function(fam, eta) {
    density <- fam$density.linkinv(eta)
    p <- density / (1 + density)
    expect_equal(fam$linkinv(eta), p, tolerance = 1e-10)
    expect_equal(fam$linkfun(p), eta, tolerance = 1e-8)
}

# Helper: numerical derivative check for mu.eta and density.eta.
check_numeric_derivs <- function(fam, eta) {
    h <- 1e-6
    mu_num <- (fam$linkinv(eta + h) - fam$linkinv(eta - h)) / (2 * h)
    density_num <- (fam$density.linkinv(eta + h) - fam$density.linkinv(eta - h)) / (2 * h)
    expect_equal(fam$mu.eta(eta), mu_num, tolerance = 1e-5)
    expect_equal(fam$density.eta(eta), density_num, tolerance = 1e-5)
}

test_that("identity link satisfies canonical property F'(density) = eta", {
    fam <- density_ratio_family("least-squares", link = "identity")
    eta <- seq(0.1, 5, length.out = 10)
    # F'(t) = t for least-squares, so F'(density.linkinv(eta)) == eta
    expect_equal(fam$density.linkinv(eta), eta)
    check_link_roundtrip(fam, eta)
    check_numeric_derivs(fam, eta)
})

test_that("log link satisfies canonical property F'(density) = eta for KL", {
    fam <- density_ratio_family("kullback-leibler", link = "log")
    eta <- seq(-2, 2, length.out = 10)
    # F'(t) = log(t), F'(exp(eta)) = eta
    expect_equal(log(fam$density.linkinv(eta)), eta)
    check_link_roundtrip(fam, eta)
    check_numeric_derivs(fam, eta)
})

test_that("inverse link satisfies F'(density) = -eta for Itakura-Saito", {
    fam <- density_ratio_family("itakura-saito", link = "inverse")
    eta <- seq(0.1, 5, length.out = 10)
    # F'(t) = -1/t, F'(1/eta) = -eta
    expect_equal(-1 / fam$density.linkinv(eta), -eta)
    check_link_roundtrip(fam, eta)
    check_numeric_derivs(fam, eta)
})

test_that("bose link satisfies F'(density) = -eta for negative-binomial", {
    fam <- density_ratio_family("negative-binomial", link = "bose")
    eta <- seq(0.1, 3, length.out = 10)
    # F'(t) = log(t/(1+t)); with t = 1/(exp(eta) - 1) we get log(1/exp(eta)) = -eta
    density <- fam$density.linkinv(eta)
    expect_equal(log(density / (1 + density)), -eta, tolerance = 1e-10)
    check_link_roundtrip(fam, eta)
    check_numeric_derivs(fam, eta)
})

test_that("softplus link approximates identity for large sharpness and positive eta", {
    # density.linkinv(eta) = softplus(s * eta) / s is a smooth max(0, eta),
    # so it converges to the identity link for positive eta as s -> infinity.
    fam <- density_ratio_family("least-squares", link = "softplus", sharpness = 20)
    eta <- seq(0.5, 3, length.out = 10)
    expect_equal(fam$density.linkinv(eta), eta, tolerance = 1e-3)
    check_link_roundtrip(fam, eta)
    check_numeric_derivs(fam, eta)
})

test_that("softplus link keeps the density strictly positive for negative eta", {
    fam <- density_ratio_family("kullback-leibler", link = "softplus", sharpness = 1)
    eta <- c(-5, -1, 0, 1, 5)
    expect_true(all(fam$density.linkinv(eta) > 0))
    check_link_roundtrip(fam, eta)
    check_numeric_derivs(fam, eta)
})
