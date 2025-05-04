NNETARCH
================

<!-- README.md is generated from README.Rmd. Please edit that file -->

# <img src="man/figures/logo.png" align="right" height="120"/>

**NNETARCH: Neural Network Autoregressive Conditional Heteroskedasticity
for Time Series Forecasting**

NNETARCH provides a hybrid forecasting framework that enhances both
point forecast accuracy and directional responsiveness by combining
nonlinear autoregressive modeling via neural networks (`nnetar`) with
time-varying volatility modeling.

The volatility component can be modeled either by a second neural
network (`nnetar`) or by a hybrid strategy that integrates GARCH(1,1)
variance estimates, using the `rugarch` package, into a neural network
structure. This unified architecture allows NNETARCH to adapt forecasts
dynamically during volatility regime shifts, extending the `forecast`
package to support variance-adaptive modeling of financial and economic
time series.

### Confidence Intervals

`forecast.nnetarch()` provides two styles of prediction intervals:

- **Forecast-style intervals** (default): Matches the `forecast` package
  by accumulating forecast uncertainty.
- **GARCH-style intervals** (`garch_confint = TRUE`): Uses direct
  conditional standard deviation from the volatility model, suitable for
  ARCH/GARCH modeling.

Users can switch using the `garch_confint` argument.

------------------------------------------------------------------------

## Installation

Install the development version of NNETARCH from GitHub:

``` r
# install.packages("devtools")
devtools::install_github("rafmontano/NNETARCH")
```

------------------------------------------------------------------------

## Example 1

``` r
library(forecast)
library(NNETARCH)

fit <- nnetarch(lynx, h = 14)
fct <- forecast(fit)
plot(fct)
```

### Forecast Output

![](man/figures/lynx_nnetarch.png)

------------------------------------------------------------------------

## Example 2

``` r
library(forecast)
library(NNETARCH)

fit <- nnetarch(lynx, h = 14, volatility.model = "garch_nnetar")
fct <- forecast(fit) # Default confidence interval style
plot(fct)

# Optionally use GARCH-style confidence intervals
fct_garch <- forecast(fit, garch_confint = TRUE)
plot(fct_garch)
```

### Forecast Output

![](man/figures/lynx_garch_nnetarch.png)

------------------------------------------------------------------------

## Example 3

``` r
library(forecast)
library(NNETARCH)
library(rugarch)

# Define a custom GARCH(1,1) specification
garch_spec_custom <- list(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "std"  # Student-t distribution for heavier tails
)

# Fit the NNETARCH model with the custom GARCH specification
fit_custom <- nnetarch(lynx, h = 14, volatility.model = "garch_nnetar", garch.control = garch_spec_custom)

# Forecast
fct_custom <- forecast(fit_custom)

# Plot
plot(fct_custom)
```

### Forecast Output

![](man/figures/lynx_garch_nnetarch_custom.png)

------------------------------------------------------------------------

## The NNETARCH framework - Architecture

The NNETARCH Framework - Architecture

The NNETARCH framework captures both nonlinear trend and conditional
volatility through a two-stage modeling architecture:

Two variants are supported:

• Model 1: Trend and volatility both modeled with neural networks.

• Model 2: Trend modeled with a neural network; volatility first
estimated via GARCH, then passed to a neural network.

![](man/figures/nnetarch_figure1-2.png)

------------------------------------------------------------------------

## Model Description

The NNETARCH model is defined as:

`y_t = f(y_{t-1}, ..., y_{t-p}) + g(e_{t-1}, ..., e_{t-q}) * ε_t`, where
`ε_t ~ N(0,1)`

Where:

- `f(.)`: Neural network for nonlinear trend  
- `g(.)`: Volatility model (either neural network on residuals, or
  GARCH-enhanced neural network)  
- `ε_t`: White noise innovation

This hybrid structure enhances the ability to forecast both the
conditional mean and variance dynamically, crucial for financial time
series.

## License

MIT © Rafael Montano, University of Sydney

------------------------------------------------------------------------
