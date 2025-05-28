# NNETARCH News

# NNETARCH 0.0.4 (Released 2025-05-28)

## Title

Forecasting with Confidence: NNETARCH for Neural Trend and Volatility-Aware Financial Prediction

## Summary

This release formalises the NNETARCH framework, a hybrid time series forecasting model that integrates nonlinear neural autoregressive trend modeling with volatility-aware adjustments based on conditional log-variance estimation.

## Key Enhancements

-   Hybrid architecture combining:
    -   A neural network (NNETAR) for modeling the nonlinear conditional mean component $f(\cdot)$
    -   A second model $h(\cdot)$ for estimating the conditional log-variance, either learned fully via NNETAR or enhanced with GARCH-based inputs
-   Probabilistic forecast generation via the `forecast.nnetarch()` method, supporting:
    -   Forecast-style prediction intervals (accumulated variance)
    -   GARCH-style prediction intervals (per-step conditional variance), configurable via `garch_confint`
-   Two supported instantiations:
    -   Model 1: Fully neural configuration (NNETAR for both mean and log-variance)
    -   Model 2: GARCH-informed volatility modeling followed by neural refinement
-   Aligned with the theoretical formulation presented in the paper titled: *Forecasting with Confidence: NNETARCH for Neural Trend and Volatility-Aware Financial Prediction*

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
