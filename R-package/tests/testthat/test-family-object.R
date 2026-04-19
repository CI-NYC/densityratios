test_that("density_ratio_family returns a stats::family object", {
    fam <- density_ratio_family("kullback-leibler")
    expect_s3_class(fam, "family")
    expect_named(
        fam,
        c(
            "family", "link",
            "linkfun", "linkinv", "mu.eta", "density.eta",
            "valideta", "density.linkinv",
            "variance", "dev.resids",
            "initialize", "validmu",
            "dispersion", "aic"
        ),
        ignore.order = TRUE
    )
})

test_that("unknown families and links error informatively", {
    expect_error(density_ratio_family("bogus"), "family 'bogus'")
    expect_error(
        density_ratio_family("kullback-leibler", link = "bogus"),
        "link 'bogus'"
    )
    expect_error(
        density_ratio_family("kullback-leibler", sharpness = -1),
        "sharpness"
    )
})
