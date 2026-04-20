#' Family object for density-ratio GLMs (Bregman--Riesz regression)
#'
#' Constructs a \code{\link[stats]{family}} object to be used in
#' density-ratio learning via Bregman--Riesz regression
#' (Hines and Miles, 2025).
#'
#' @param family Character, one of \code{"least-squares"},
#'   \code{"kullback-leibler"}, \code{"itakura-saito"} or
#'   \code{"negative-binomial"}. Selects the Bregman divergence to be minimized.
#' @param link Character link function mapping the linear predictor
#'   \eqn{\eta} to the density ratio \eqn{\alpha}. One of
#'   \code{"identity"}, \code{"log"}, \code{"inverse"}, \code{"bose"} or
#'   \code{"softplus"}. When \code{NULL}, the canonical link for the
#'   chosen family is used. See Details for the per-link formulas.
#' @param sharpness Positive scalar controlling the sharpness of the
#'   softplus link. Only used when \code{link = "softplus"}; see Details.
#'
#' @return An S3 object inheriting from \code{"family"} suitable for use as
#'   the \code{family} argument of, e.g. \code{\link[stats]{glm}}. In addition to
#'   the usual \code{family} components, the returned object contains
#'   \describe{
#'     \item{\code{density.linkinv}}{Maps the linear predictor \eqn{\eta} to
#'       the density-ratio scale.}
#'     \item{\code{density.eta}}{The derivative of \code{density.linkinv}
#'       with respect to \eqn{\eta}.}
#'   }
#'   After fitting, e.g. \code{glm} with this family, density-ratio predictions
#'   are obtained as \code{family$density.linkinv(predict(model, type = "link"))}.
#'
#' @details
#' Let \eqn{p = \alpha/(1 + \alpha)} be the density ratio \eqn{\alpha}
#' expressed on a probability scale. Under the pseudo-outcome
#' reformulation, minimising the empirical Bregman--Riesz risk is equivalent
#' to regressing a pseudo-outcome \eqn{q_i \in [0, 1]} and variance
#' function \eqn{V(p) = (1 - p)^3 / F''(p/(1-p))}. The family objects
#' returned here implement \eqn{V(p)} for each divergence:
#' \tabular{ll}{
#'   least-squares \tab \eqn{V(p) = (1-p)^3} \cr
#'   Kullback--Leibler \tab \eqn{V(p) = p(1-p)^2} \cr
#'   Itakura--Saito \tab \eqn{V(p) = p^2(1-p)} \cr
#'   negative-binomial \tab \eqn{V(p) = p(1-p)} \cr
#' }
#'
#' \strong{Link functions.} Included are the link functions:
#' \describe{
#'   \item{\code{"identity"}}{\eqn{\alpha = \eta}. Requires
#'     \eqn{\eta > 0} because the density ratio is non-negative.
#'     Canonical link for least-squares.}
#'   \item{\code{"log"}}{\eqn{\alpha = \exp(\eta)}. The linear predictor
#'     is the log density ratio, which is unconstrained; Canonical link for Kullback--Leibler.}
#'   \item{\code{"inverse"}}{\eqn{\alpha = 1 / \eta}. Requires
#'     \eqn{\eta > 0}; \eqn{\eta} is then the \emph{inverse} density
#'     ratio. Canonical link for Itakura--Saito.}
#'   \item{\code{"bose"}}{\eqn{\alpha = 1 / (e^\eta - 1)}, named after the
#'     Bose--Einstein occupation number. Requires \eqn{\eta > 0}.
#'     Canonical link for negative-binomial.}
#'   \item{\code{"softplus"}}{\eqn{\alpha = \log(1 + e^{s \eta}) / s} with
#'     positive sharpness \eqn{s = }\code{sharpness}. A smooth approximation of
#'     \eqn{\max(0, \eta)}. The linear predictor is unconstrained.}
#' }
#'
#' @references
#' Hines, O. J. and Miles, C. H. (2025).
#' \emph{Learning density ratios in causal inference using
#' Bregman--Riesz regression.}
#'
#' @examples
#' set.seed(42)
#' n <- 200
#' df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
#' numerator_w <- runif(n, 0.8, 1.2)
#' denominator_w <- runif(n, 0.8, 1.2)
#' w <- numerator_w + denominator_w
#' df$pseudo_outcome <- numerator_w / w
#' pseudo_weights <- w / 2
#'
#' fam <- density_ratio_family("kullback-leibler")
#' model <- glm(
#'     pseudo_outcome ~ x1 + x2,
#'     weights = pseudo_weights,
#'     family = fam,
#'     data = df
#' )
#'
#' # Density-ratio predictions on the original scale
#' eta <- predict(model, type = "link")
#' density_ratio <- fam$density.linkinv(eta)
#'
#' @export
density_ratio_family <- function(
    family = "negative-binomial",
    link = NULL,
    sharpness = 1
) {
    families <- c(
        "least-squares",
        "kullback-leibler",
        "itakura-saito",
        "negative-binomial"
    )
    if (!family %in% families) {
        stop(sprintf(
            "family '%s' not recognized; choose one of %s.",
            family, paste(sQuote(families), collapse = ", ")
        ))
    }
    if (!is.numeric(sharpness) || length(sharpness) != 1L || sharpness <= 0) {
        stop("'sharpness' must be a single positive numeric value.")
    }

    # Canonical links: density.linkinv(eta) = g^{-1}(eta) is defined so that
    # F'{density.linkinv(eta)} = +/- eta. The sign is chosen for convenience,
    # with the negative sign used when density.linkinv requires eta > 0.
    canonical_links <- list(
        `least-squares` = "identity",
        `kullback-leibler` = "log",
        `itakura-saito` = "inverse",
        `negative-binomial` = "bose"
    )
    if (is.null(link)) {
        link <- canonical_links[[family]]
    }

    linkstats <- link_stats(link, sharpness)
    familystats <- family_stats(family)

    structure(
        list(
            family = family,
            link = link,
            linkfun = linkstats$linkfun,
            linkinv = linkstats$linkinv,
            mu.eta = linkstats$mu.eta,
            density.eta = linkstats$density.eta,
            valideta = linkstats$valideta,
            density.linkinv = linkstats$density.linkinv,
            variance = familystats$variance,
            dev.resids = familystats$dev.resids,
            initialize = expression({
                n <- rep.int(1, nobs)
                mustart <- rep(0.5, nobs)
            }),
            validmu = function(mu) all(is.finite(mu)) && all(0 < mu & mu < 1),
            dispersion = NA_real_,
            aic = function(y, n, mu, wt, dev) NA_real_
        ),
        class = "family"
    )
}

# Finite nonnegative eta validator, shared by links with density > 0.
finite_non_negative <- function(eta) {
    all(is.finite(eta)) && all(eta > 0)
}

# Link ingredients: map linear predictor eta to density-ratio and
# probability scales, with first derivatives.
link_stats <- function(link, sharpness) {
    if (link == "identity") {
        list(
            density.linkinv = identity,
            linkinv = function(eta) eta / (1 + eta),
            linkfun = function(p) p / (1 - p),
            mu.eta = function(eta) (1 + eta)^(-2),
            density.eta = function(eta) rep.int(1, length(eta)),
            valideta = finite_non_negative
        )
    } else if (link == "log") {
        qb <- stats::quasibinomial()
        list(
            density.linkinv = exp,
            linkinv = qb$linkinv,
            linkfun = qb$linkfun,
            mu.eta = qb$mu.eta,
            density.eta = exp,
            valideta = qb$valideta
        )
    } else if (link == "inverse") {
        list(
            density.linkinv = function(eta) 1 / eta,
            linkinv = function(eta) 1 / (1 + eta),
            linkfun = function(p) (1 / p) - 1,
            mu.eta = function(eta) -(1 + eta)^(-2),
            density.eta = function(eta) -eta^(-2),
            valideta = finite_non_negative
        )
    } else if (link == "bose") {
        list(
            density.linkinv = function(eta) 1 / expm1(eta),
            linkinv = function(eta) exp(-eta),
            linkfun = function(p) -log(p),
            mu.eta = function(eta) -exp(-eta),
            density.eta = function(eta) -exp(eta) / expm1(eta)^2,
            valideta = finite_non_negative
        )
    } else if (link == "softplus") {
        qb <- stats::quasibinomial()
        softplus <- function(x) log1p(exp(x))
        list(
            density.linkinv = function(eta) softplus(sharpness * eta) / sharpness,
            linkinv = function(eta) {
                dr <- softplus(sharpness * eta) / sharpness
                dr / (1 + dr)
            },
            linkfun = function(p) {
                dr <- p / (1 - p)
                log(expm1(sharpness * dr)) / sharpness
            },
            mu.eta = function(eta) {
                density <- softplus(sharpness * eta) / sharpness
                qb$linkinv(sharpness * eta) / (density + 1)^2
            },
            density.eta = function(eta) qb$linkinv(sharpness * eta),
            valideta = function(eta) all(is.finite(eta))
        )
    } else {
        stop(sprintf("link '%s' not recognized.", link))
    }
}

# Bregman divergence on the probability scale:
#   variance V(p) = (1 - p)^3 / F''(p/(1-p))
#   dev.resids    = 2 * wt * {-F_tilde(mu) - F_tilde'(mu)(y - mu)}
# where F_tilde(t) = (1 - t) F(t/(1 - t))
family_stats <- function(family) {
    if (family == "least-squares") {
        list(
            variance = function(p) (1 - p)^3,
            dev.resids = function(y, mu, wt) {
                wt * (mu^2 * (1 + y) - 2 * mu * y) / (1 - mu)^2
            }
        )
    } else if (family == "kullback-leibler") {
        list(
            variance = function(p) p * (1 - p)^2,
            dev.resids = function(y, mu, wt) {
                odds <- mu / (1 - mu)
                dev <- odds * (1 - y) - y * log(odds)
                2 * wt * dev
            }
        )
    } else if (family == "itakura-saito") {
        list(
            variance = function(p) p^2 * (1 - p),
            dev.resids = function(y, mu, wt) {
                neg_odds <- (1 - mu) / mu
                dev <- y * neg_odds - (1 - y) * log(neg_odds)
                2 * wt * dev
            }
        )
    } else if (family == "negative-binomial") {
        qb <- stats::quasibinomial()
        list(variance = qb$variance, dev.resids = qb$dev.resids)
    } else {
        stop(sprintf("family '%s' not recognized.", family))
    }
}
