#' NNETARCH: Neural Network Autoregressive Model with Conditional Volatility
#'
#' Fits the NNETARCH model—a hybrid framework combining a nonlinear autoregressive neural network (NNETAR)
#' for trend forecasting with a second module that models conditional volatility through a log-variance function.
#' Volatility estimation can be purely neural (Model 1) or GARCH-informed (Model 2).
#'
#' @param y A univariate time series object of class \code{ts}.
#' @param volatility Logical. If FALSE, only the trend component is fitted. Default is TRUE.
#' @param volatility.model Either \code{"nnetar"} (Model 1: Fully Neural Volatility) or \code{"garch_nnetar"} (Model 2: GARCH-Informed Volatility).
#' @param garch.control Optional. A named list of parameters passed to \code{rugarch::ugarchspec()} when using GARCH-based volatility modeling.
#' @param ... Additional arguments passed to \code{\link[forecast]{nnetar}}.
#'
#' @return An object of class \code{"nnetarch"} with the following components:
#' \describe{
#'   \item{\code{trend_model}}{Fitted NNETAR model for the conditional mean \( f(\cdot) \)}
#'   \item{\code{vol_model}}{Volatility model \( h(\cdot) \): either a second NNETAR or GARCH-enhanced neural model}
#'   \item{\code{x}}{Original input time series}
#'   \item{\code{method}}{String label: \code{"NNETARCH"}}
#'   \item{\code{volatility_model_type}}{Type of volatility modeling used}
#'   \item{\code{call}}{Original function call}
#' }
#'
#' @details
#' The trend \( f(\cdot) \) is estimated using a standard NNETAR model on lagged inputs. Residuals are then used to model log-variance \( h(\cdot) \):
#' \itemize{
#'   \item \strong{Model 1 ("nnetar")}: A second NNETAR is trained on squared residuals \( e_t^2 \) to estimate volatility.
#'   \item \strong{Model 2 ("garch_nnetar")}: Conditional variances \( \hat{\sigma}^2_t \) from a GARCH(1,1) model are used alongside residuals as inputs to train a second NNETAR.
#' }
#' Forecasts are then synthesized as:
#' \deqn{
#' y_{t+h} = f(y_{t-1}, ..., y_{t-p}) + \exp(0.5 \cdot h(\cdot)) \cdot \varepsilon_{t+h}, \quad \varepsilon_{t+h} \sim \mathcal{N}(0, 1)
#' }
#'
#' @export
nnetarch <- function(y, volatility = TRUE,
                     volatility.model = c("nnetar", "garch_nnetar"),
                     garch.control = NULL,
                     ...) {

  if (!inherits(y, "ts")) stop("Input y must be a time series object (ts)")

  volatility.model <- match.arg(volatility.model)

  # Step 1: Trend Estimation using NNETAR
  trend_model <- forecast::nnetar(y, ...)
  residuals <- stats::residuals(trend_model)
  vol_model <- NULL

  # Step 2: Volatility Estimation (Conditional Log-Variance)
  if (volatility) {
    if (volatility.model == "nnetar") {
      # Model 1: Fully Neural Volatility
      res_sq <- residuals^2
      res_sq <- res_sq[is.finite(res_sq)]
      vol_model <- forecast::nnetar(res_sq, ...)

    } else if (volatility.model == "garch_nnetar") {
      # Model 2: GARCH-Informed Neural Volatility
      if (!requireNamespace("rugarch", quietly = TRUE)) {
        stop("Please install the 'rugarch' package to use GARCH-based volatility modeling.")
      }

      garch_spec <- if (is.null(garch.control)) {
        rugarch::ugarchspec(
          variance.model = list(garchOrder = c(1, 1)),
          mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
          distribution.model = "norm"
        )
      } else {
        do.call(rugarch::ugarchspec, garch.control)
      }

      residuals_clean <- na.omit(residuals)
      garch_fit <- tryCatch(
        rugarch::ugarchfit(spec = garch_spec, data = residuals_clean),
        error = function(e) {
          warning("GARCH fitting failed: ", e$message)
          return(NULL)
        }
      )

      if (is.null(garch_fit) || garch_fit@fit$convergence != 0) {
        warning("Proceeding without volatility modeling due to GARCH fitting issue.")
        vol_model <- NULL
      } else {
        garch_forecast <- rugarch::ugarchforecast(garch_fit, n.ahead = length(residuals_clean))
        garch_variances <- as.numeric(rugarch::sigma(garch_forecast))^2
        inputs_mat <- cbind(residuals_clean, garch_variances)
        vol_model <- forecast::nnetar(as.numeric(inputs_mat), ...)
      }
    }
  }

  # Step 3: Output Structured Object
  structure(list(
    method = "NNETARCH",
    x = y,
    trend_model = trend_model,
    vol_model = vol_model,
    volatility_model_type = volatility.model,
    call = match.call()
  ), class = "nnetarch")
}

#' Forecast Method for NNETARCH Objects
#'
#' Generates probabilistic forecasts from a fitted \code{nnetarch} model by combining a nonlinear trend forecast with an uncertainty term derived from a learned log-variance model. Supports both cumulative variance and per-step conditional variance prediction intervals.
#'
#' @param object An object of class \code{nnetarch}.
#' @param h Number of forecast steps ahead.
#' @param level Numeric vector of confidence levels for prediction intervals. Default is \code{c(80, 95)}.
#' @param garch_confint Logical. If \code{TRUE}, prediction intervals use conditional variance (GARCH-style); if \code{FALSE}, intervals accumulate forecast variance over horizon. Default is \code{FALSE}.
#' @param ... Additional arguments passed to the underlying \code{forecast()} calls.
#'
#' @return An object of class \code{forecast}, including forecast means, prediction intervals, fitted values, and residuals.
#'
#' @method forecast nnetarch
#' @export
#' @importFrom stats residuals
forecast.nnetarch <- function(object, h = 10, level = c(80, 95), garch_confint = FALSE, ...) {
  # Forecast trend
  trend_fc <- forecast::forecast(object$trend_model, h = h, level = level, ...)
  mean_fc <- as.numeric(trend_fc$mean)

  # If no volatility model, return trend forecast
  if (is.null(object$vol_model) && is.null(object$logvar_model)) {
    return(trend_fc)
  }

  # Determine volatility model
  vol_model <- if (!is.null(object$logvar_model)) object$logvar_model else object$vol_model

  # Forecast log-variance (logvar_model) or variance (vol_model)
  vol_fc <- forecast::forecast(vol_model, h = h, ...)
  vol_raw <- as.numeric(vol_fc$mean)

  # Convert to standard deviation
  sigma <- if (!is.null(object$logvar_model)) sqrt(exp(vol_raw)) else sqrt(pmax(vol_raw, 1e-6))

  # Generate noise
  set.seed(123)
  epsilon <- rnorm(h)
  vol_term <- sigma * epsilon

  # Combine
  hybrid_mean <- ts(mean_fc + vol_term,
                    start = start(trend_fc$mean),
                    frequency = frequency(trend_fc$mean))

  # Confidence intervals
  level <- sort(level)
  z_vals <- qnorm(1 - (1 - level / 100) / 2)

  if (garch_confint) {
    lower <- sapply(z_vals, function(z) mean_fc - z * sigma)
    upper <- sapply(z_vals, function(z) mean_fc + z * sigma)
  } else {
    sigma_accum <- sqrt(cumsum(sigma^2))
    lower <- sapply(z_vals, function(z) mean_fc - z * sigma_accum)
    upper <- sapply(z_vals, function(z) mean_fc + z * sigma_accum)
  }

  lower_ts <- ts(lower, start = start(trend_fc$mean), frequency = frequency(trend_fc$mean))
  upper_ts <- ts(upper, start = start(trend_fc$mean), frequency = frequency(trend_fc$mean))

  # Return forecast object
  structure(list(
    method = "NNETARCH",
    model = object,
    mean = hybrid_mean,
    lower = lower_ts,
    upper = upper_ts,
    level = level,
    x = object$x,
    fitted = fitted(object$trend_model),
    residuals = stats::residuals(object$trend_model)
  ), class = "forecast")
}
