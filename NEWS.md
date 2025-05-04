# NNETARCH News

## Version 0.0.3 (2025-05-05)

-   This release is dedicated to the memory of Francisco Jose Avila Fuenmayor ("Franco").
-   Added support for dual confidence interval schemes in `forecast.nnetarch()`:
    -   Standard `forecast`-style intervals (default).
    -   Optional `garch_confint = TRUE` for GARCH-style intervals based on conditional volatility.
-   Updated `README.Rmd` and examples to reflect new `garch_confint` argument.
-   Improved documentation to clarify interpretation of confidence intervals in hybrid volatility modeling.

## Version 0.0.2 (Released 2025-04-28)

-   Added support for GARCH-enhanced volatility modeling (`volatility.model = "garch_nnetar"`), using the `rugarch` package.
-   Improved error handling for GARCH fitting (graceful fallback if convergence fails).
-   Updated documentation and README to reflect new architectures and examples.
-   Minor internal cleanups and refactoring.

## Version 0.0.1 (Released 2025-04-21)

-   Initial release.
-   Implemented hybrid NNETAR + volatility model based on squared residuals.
-   Basic integration with the `forecast` package for seamless forecasting.
