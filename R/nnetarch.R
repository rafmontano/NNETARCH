#' Neural Network Autoregressive Model with Conditional Volatility
#'
#' Fits a hybrid forecasting model that combines a nonlinear autoregressive model using a neural network (NNETAR) for the conditional mean with a second model for conditional volatility. The volatility component can be either a second NNETAR model or a GARCH(1,1) model via the \pkg{rugarch} package.
#'
#' @param y A univariate time series of class \code{ts}.
#' @param h Forecast horizon (number of steps ahead).
#' @param volatility.model A string indicating the volatility modeling method. Either \code{"nnetar"} (default) or \code{"garch"}.
#' @param p Number of non-seasonal lags for the NNETAR trend model (optional).
#' @param q Reserved for future use (currently ignored).
#' @param size Number of hidden nodes in the NNETAR trend model.
#' @param repeats Number of repetitions for the NNETAR model with different random initializations (default: 20).
#' @param lambda Box-Cox transformation parameter. If \code{"auto"}, it is selected automatically.
#' @param simulate Logical. If \code{TRUE} (default), standard normal noise is added to the forecast via the volatility path.
#' @param seed Random seed used when simulating \code{simulate = TRUE}.
#' @param garch.control A named list of parameters passed to \code{ugarchspec()} when using the \pkg{rugarch} volatility backend.
#' @param ... Additional arguments passed to the underlying \code{\link[forecast]{nnetar}} function.
#'
#' @return An object of class \code{"nnetarch"} containing the following elements:
#' \describe{
#'   \item{\code{f_model}}{Fitted trend model from \code{nnetar()}}
#'   \item{\code{g_model}}{Fitted volatility model (either \code{nnetar} or \code{ugarchforecast})}
#'   \item{\code{mean}}{Hybrid forecast values combining trend and simulated volatility}
#'   \item{\code{residuals}}{Residuals from the trend model}
#'   \item{\code{fitted}}{Fitted values from the trend model}
#'   \item{\code{inputs}}{List of original inputs used in fitting the model}
#' }
#'
#' @details
#' The model forecasts the trend using a neural network autoregression (\code{nnetar}). The residuals from this model are then used to estimate volatility either by:
#' \itemize{
#'   \item Fitting a second neural network model to squared residuals, or
#'   \item Fitting a GARCH(1,1) model to the residuals using the \pkg{rugarch} package.
#' }
#' The final forecast is computed as:
#' \deqn{
#' \hat{y}_{t+h} = f(y_{t-1}, \dots, y_{t-p}) + g(\cdot) \cdot \varepsilon_t
#' }
#' where \eqn{\varepsilon_t \sim \mathcal{N}(0, 1)} is simulated white noise and \eqn{g(\cdot)} is a forecast of the conditional standard deviation.
#'
#' @seealso \code{\link[forecast]{nnetar}}, \code{\link[forecast]{forecast}}, \code{\link[rugarch]{ugarchfit}}, \code{\link{forecast.nnetarch}}
#'
#' @examples
#' library(forecast)
#' fit <- nnetarch(lynx, h = 14)
#' plot(forecast(fit))
#'
#' @return An object of class "nnetarch".
#' @importFrom forecast forecast
#' @importFrom stats residuals
#' @export
nnetarch <- function(y, volatility = TRUE, ...) {
  if (!inherits(y, "ts")) stop("Input y must be a time series object (ts)")

  # Step 1: Trend model using NNETAR
  trend_model <- forecast::nnetar(y, ...)
  residuals <- residuals(trend_model)

  # Step 2: Volatility model on squared residuals
  vol_model <- NULL
  if (volatility) {
    res_sq <- residuals^2
    res_sq <- res_sq[is.finite(res_sq)]
    vol_model <- forecast::nnetar(res_sq)
  }

  structure(list(
    method = "NNETARCH",
    x = y,
    trend_model = trend_model,
    vol_model = vol_model,
    call = match.call()
  ), class = "nnetarch")
}

#' Forecast Method for NNETARCH Objects
#'
#' Generates forecasts from an object of class \code{nnetarch}, which combines a NNETAR trend model and a volatility model.
#'
#' @param object An object of class \code{nnetarch}.
#' @param h Number of steps ahead for forecasting.
#' @param ... Additional arguments passed to the underlying \code{forecast()} function.
#'
#' @return An object of class \code{forecast} containing the hybrid forecast.
#' @method forecast nnetarch
#' @export
forecast.nnetarch <- function(object, h = 10, level = c(80, 95), ...) {
  # Step 1: Forecast the trend component
  trend_fc <- forecast::forecast(object$trend_model, h = h, level = level, ...)

  # Step 2: Forecast the volatility component (if present)
  if (!is.null(object$vol_model)) {
    vol_fc <- forecast::forecast(object$vol_model, h = h, ...)

    # Simulate standard normal noise
    set.seed(123)
    epsilon <- rnorm(h, mean = 0, sd = 1)

    # Calculate volatility term
    vol_term <- sqrt(vol_fc$mean) * epsilon
    vol_term_ts <- ts(vol_term,
                      start = start(trend_fc$mean),
                      frequency = frequency(trend_fc$mean))

    # Hybrid forecast: combine trend and volatility
    hybrid_mean <- ts(trend_fc$mean + vol_term_ts,
                      start = start(trend_fc$mean),
                      frequency = frequency(trend_fc$mean))
  } else {
    hybrid_mean <- trend_fc$mean
    vol_fc <- NULL
  }

  # Step 3: Compute prediction intervals using volatility estimates
  sigma_h <- if (!is.null(vol_fc)) sqrt(vol_fc$mean) else rep(0, h)
  mean_fc <- as.numeric(trend_fc$mean)
  level <- sort(level)
  z_vals <- qnorm(1 - (1 - level / 100) / 2)

  lower <- sapply(z_vals, function(z) mean_fc - z * sigma_h)
  upper <- sapply(z_vals, function(z) mean_fc + z * sigma_h)

  # Convert to time series objects
  lower_ts <- ts(lower, start = start(trend_fc$mean), frequency = frequency(trend_fc$mean))
  upper_ts <- ts(upper, start = start(trend_fc$mean), frequency = frequency(trend_fc$mean))

  # Step 4: Return a valid forecast object (class = "forecast")
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

