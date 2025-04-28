# NNETARCH News

## Version 0.0.2 (Released 2025-04-28)

-   Added support for GARCH-enhanced volatility modeling (`volatility.model = "garch_nnetar"`), using the `rugarch` package.
-   Improved error handling for GARCH fitting (graceful fallback if convergence fails).
-   Updated documentation and README to reflect new architectures and examples.
-   Minor internal cleanups and refactoring.

## Version 0.0.1 (Released 2025-04-21)

-   Initial release.
-   Implemented hybrid NNETAR + volatility model based on squared residuals.
-   Basic integration with the `forecast` package for seamless forecasting.
