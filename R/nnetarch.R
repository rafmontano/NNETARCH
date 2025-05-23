#' Neural Network Autoregressive Model with Conditional Volatility
#'
#' Fits a hybrid forecasting model that combines a nonlinear autoregressive model using a neural network (NNETAR) for the conditional mean with a second model for conditional volatility. The volatility component can be either a second NNETAR model or a GARCH-enhanced NNETAR model via the \pkg{rugarch} package.
#'
#' @param y A univariate time series of class \code{ts}.
#' @param volatility Logical. If FALSE, only trend estimation is performed. Default is TRUE.
#' @param volatility.model Character. "nnetar" for NNETAR residual volatility (Model 1), or "garch_nnetar" for GARCH-enhanced NNETAR modeling (Model 2). Default is "nnetar".
#' @param garch.control A named list of parameters passed to \code{ugarchspec()} when using the \pkg{rugarch} volatility backend. Optional.
#' @param ... Additional arguments passed to the underlying \code{\link[forecast]{nnetar}} function for trend modeling.
#'
#' @return An object of class \code{"nnetarch"} containing:
#' \describe{
#'   \item{\code{trend_model}}{Fitted trend model from \code{nnetar()}}
#'   \item{\code{vol_model}}{Fitted volatility model (either \code{nnetar} or GARCH-enhanced \code{nnetar})}
#'   \item{\code{x}}{Original input series}
#'   \item{\code{method}}{Description of the method used}
#'   \item{\code{volatility_model_type}}{The volatility model used ("nnetar" or "garch_nnetar")}
#'   \item{\code{call}}{The original function call}
#' }
#'
#' @details
#' The NNETARCH model forecasts the trend using a neural network autoregression (\code{nnetar}). The residuals from this model are then used to estimate volatility through two alternatives:
#' \itemize{
#'   \item \strong{Model 1 (nnetar):} A second neural network is fitted to the squared residuals to model the variance dynamics.
#'   \item \strong{Model 2 (garch_nnetar):} Conditional variances are first estimated using a GARCH(1,1) model, and these are then used alongside residuals to train a second neural network.
#' }
#' The final forecast is computed as:
#' \deqn{
#' \hat{y}_{t+h} = f(y_{t-1}, \dots, y_{t-p}) + g(\cdot) \cdot \varepsilon_t
#' }
#' where \eqn{\varepsilon_t \sim \mathcal{N}(0, 1)} is simulated white noise, and \eqn{g(\cdot)} is a learned function estimating the conditional standard deviation.
#'
#' @seealso \code{\link[forecast]{nnetar}}, \code{\link[forecast]{forecast}}, \code{\link[rugarch]{ugarchfit}}, \code{\link{forecast.nnetarch}}
#'
#' @examples
#' library(forecast)
#' fit <- nnetarch(lynx)
#' plot(forecast(fit, h = 14))
#'
#' @importFrom forecast forecast
#' @importFrom stats residuals
#' @export
nnetarch <- function(y, volatility = TRUE,
                     volatility.model = c("nnetar", "garch_nnetar"),
                     garch.control = NULL,
                     ...) {

  if (!inherits(y, "ts")) stop("Input y must be a time series object (ts)")

  volatility.model <- match.arg(volatility.model)

  # Step 1: Trend model using NNETAR
  trend_model <- forecast::nnetar(y, ...)
  residuals <- stats::residuals(trend_model)

  # Step 2: Volatility model
  vol_model <- NULL

  if (volatility) {
    if (volatility.model == "nnetar") {
      # ✅ Original behavior
      res_sq <- residuals^2
      res_sq <- res_sq[is.finite(res_sq)]
      vol_model <- forecast::nnetar(res_sq, ...)

    } else if (volatility.model == "garch_nnetar") {
      # 🆕 New behavior: GARCH-enhanced NNETAR
      if (!requireNamespace("rugarch", quietly = TRUE)) {
        stop("Package 'rugarch' must be installed for volatility.model = 'garch_nnetar'")
      }

      if (is.null(garch.control)) {
        garch_spec <- rugarch::ugarchspec(
          variance.model = list(garchOrder = c(1, 1)),
          mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
          distribution.model = "norm"
        )
      } else {
        garch_spec <- do.call(rugarch::ugarchspec, garch.control)
      }

      # 🛡️ Safely attempt to fit GARCH
      garch_fit <- tryCatch(
        {
          residuals_clean <- na.omit(residuals)
          rugarch::ugarchfit(spec = garch_spec, data = residuals_clean)
        },
        error = function(e) {
          warning("GARCH fitting failed: ", e$message)
          return(NULL)
        }
      )

      if (is.null(garch_fit)) {
        warning("Proceeding without volatility modeling due to GARCH fitting failure.")
        vol_model <- NULL
      } else if (garch_fit@fit$convergence != 0) {
        warning("GARCH fitting did not converge. Proceeding without volatility modeling.")
        vol_model <- NULL
      } else {
        # ✅ GARCH fitted successfully
        garch_forecast <- rugarch::ugarchforecast(garch_fit, n.ahead = length(residuals_clean))
        garch_variances <- as.numeric(rugarch::sigma(garch_forecast))^2

        # Input matrix: [residuals, garch variances]
        inputs_mat <- cbind(residuals_clean, garch_variances)

        vol_model <- forecast::nnetar(as.numeric(inputs_mat), ...)
      }
    }
  }

  # Step 3: Return structure
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
#' Generates forecasts from an object of class \code{nnetarch}, combining trend and volatility forecasts.
#' The user may choose between forecast-style cumulative variance prediction intervals (default)
#' and GARCH-style conditional variance intervals.
#'
#' @param object An object of class \code{nnetarch}.
#' @param h Number of steps ahead for forecasting.
#' @param level Confidence levels for prediction intervals (numeric vector). Default is \code{c(80, 95)}.
#' @param garch_confint Logical. If \code{TRUE}, uses GARCH-style prediction intervals based on conditional volatility.
#'                       If \code{FALSE} (default), uses cumulative variance for prediction intervals as in ETS/ARIMA.
#' @param ... Additional arguments passed to the underlying \code{forecast()} methods.
#'
#' @return An object of class \code{forecast}, containing the forecast mean, intervals, and metadata.
#'
#' @examples
#' \dontrun{
#' fit <- nnetarch(y)
#' forecast(fit, h = 12)  # default intervals
#' forecast(fit, h = 12, garch_confint = TRUE)  # conditional variance intervals
#' }
#'
#' @method forecast nnetarch
#' @export
forecast.nnetarch <- function(object, h = 10, level = c(80, 95), garch_confint = FALSE, ...) {
  # Step 1: If no volatility model, defer to trend model
  if (is.null(object$vol_model)) {
    return(forecast::forecast(object$trend_model, h = h, level = level, ...))
  }

  # Step 2: Forecast the trend component
  trend_fc <- forecast::forecast(object$trend_model, h = h, level = level, ...)

  # Step 3: Forecast the volatility component
  vol_fc <- forecast::forecast(object$vol_model, h = h, ...)

  # Step 4: Simulate normal noise and scale by volatility
  set.seed(123)
  epsilon <- rnorm(h, mean = 0, sd = 1)
  vol_mean <- pmax(vol_fc$mean, 1e-6)  # prevent negatives
  vol_term <- sqrt(vol_mean) * epsilon
  vol_term_ts <- ts(vol_term, start = start(trend_fc$mean), frequency = frequency(trend_fc$mean))

  # Step 5: Combine trend and volatility
  hybrid_mean <- ts(trend_fc$mean + vol_term_ts,
                    start = start(trend_fc$mean),
                    frequency = frequency(trend_fc$mean))

  # Step 6: Compute intervals
  level <- sort(level)
  z_vals <- qnorm(1 - (1 - level / 100) / 2)
  mean_fc <- as.numeric(trend_fc$mean)

  if (garch_confint) {
    # Option 1: GARCH-style intervals (per-step conditional variance)
    sigma <- sqrt(vol_mean)
    lower <- sapply(z_vals, function(z) mean_fc - z * sigma)
    upper <- sapply(z_vals, function(z) mean_fc + z * sigma)
  } else {
    # Option 2: Forecast-style intervals (cumulative variance)
    sigma_raw <- sqrt(vol_mean)
    sigma_accum <- sqrt(cumsum(sigma_raw^2))  # like ETS/ARIMA
    lower <- sapply(z_vals, function(z) mean_fc - z * sigma_accum)
    upper <- sapply(z_vals, function(z) mean_fc + z * sigma_accum)
  }

  # Step 7: Convert to time series objects
  lower_ts <- ts(lower, start = start(trend_fc$mean), frequency = frequency(trend_fc$mean))
  upper_ts <- ts(upper, start = start(trend_fc$mean), frequency = frequency(trend_fc$mean))

  # Step 8: Return forecast object
  structure(list(
    method = "NNETARCH",
    model = object,
    mean = hybrid_mean,
    lower = lower_ts,
    upper = upper_ts,
    level = level,
    x = object$x,
    fitted = fitted(object$trend_model),
    residuals = residuals(object$trend_model)
  ), class = "forecast")
}
