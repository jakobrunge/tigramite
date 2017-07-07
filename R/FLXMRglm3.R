FLXMRglm3 <- function (formula = . ~ ., family = c("gaussian", "binomial",
                                      "poisson", "Gamma"), offset = NULL)
{
  family <- match.arg(family)

  z <- new("FLXMRglm", weighted = TRUE, formula = formula,
           name = paste("FLXMRglm", family, sep = ":"), offset = offset,
           family = family)
  z@preproc.y <- function(x) {
    #if (ncol(x) > 1)
      #stop(paste("for the", family, "family y must be univariate"))
    x
  }
  if (family == "gaussian") {
    z@defineComponent <- expression({
      predict <- function(x, ...) {
        dotarg = list(...)
        if ("offset" %in% names(dotarg)) offset <- dotarg$offset
        p <- x %*% coef
        if (!is.null(offset)) p <- p + offset
        p
      }
      logLik <- function(x, y, ...){

        if (!is.matrix(y) | ncol(y)==1){
          pred = y-predict(x, ...);
          LL = dnorm(pred, mean = 0, sd=sqrt(sigma), log = TRUE);
        } else {
          pred = y-predict(x, ...);
          LL= dmvnorm(pred, rep(0,ncol(y)), sigma, log = TRUE);
        }
        return(LL)
      }
      new("FLXcomponent", parameters = list(coef = coef,
                                            sigma = sigma), logLik = logLik, predict = predict,
          df = df)
    })
    z@fit <- function(x, y, w, component) {
      fit <- lm.wfit(x, y, w = w, offset = offset)
      resid = repmat(matrix2(sqrt(fit$weights)),1,ncol(matrix2(fit$residuals))) * fit$residuals;

      #with(list(coef = coef(fit), df = ncol(x) + 1, sigma = sqrt(sum(fit$weights *
                                                                       #fit$residuals^2/mean(fit$weights))/(nrow(x) -
                                                                                                             #fit$rank))), eval(z@defineComponent))

      #with(list(coef = coef(fit), df = ncol(x) + 1, sigma = ((t(resid)%*%resid)/mean(fit$weights)/(nrow(x) - fit$rank))), eval(z@defineComponent))
      with(list(coef = coef(fit), df = ncol(x) + 1, sigma = ((t(resid)%*%resid)/mean(fit$weights)/(nrow(x) - 1))), eval(z@defineComponent))
      }
  }
  else if (family == "binomial") {
    z@preproc.y <- function(x) {
      if (ncol(x) != 2)
        stop("for the binomial family, y must be a 2 column matrix\n",
             "where col 1 is no. successes and col 2 is no. failures")
      if (any(x < 0))
        stop("negative values are not allowed for the binomial family")
      x
    }
    z@defineComponent <- expression({
      predict <- function(x, ...) {
        dotarg = list(...)
        if ("offset" %in% names(dotarg)) offset <- dotarg$offset
        p <- x %*% coef
        if (!is.null(offset)) p <- p + offset
        get(family, mode = "function")()$linkinv(p)
      }
      logLik <- function(x, y, ...) dbinom(y[, 1], size = rowSums(y),
                                           prob = predict(x, ...), log = TRUE)
      new("FLXcomponent", parameters = list(coef = coef),
          logLik = logLik, predict = predict, df = df)
    })
    z@fit <- function(x, y, w, component) {
      fit <- glm.fit(x, y, weights = w, family = binomial(),
                     offset = offset, start = component$coef)
      with(list(coef = coef(fit), df = ncol(x)), eval(z@defineComponent))
    }
  }
  else if (family == "poisson") {
    z@defineComponent <- expression({
      predict <- function(x, ...) {
        dotarg = list(...)
        if ("offset" %in% names(dotarg)) offset <- dotarg$offset
        p <- x %*% coef
        if (!is.null(offset)) p <- p + offset
        get(family, mode = "function")()$linkinv(p)
      }
      logLik <- function(x, y, ...) dpois(y, lambda = predict(x,
                                                              ...), log = TRUE)
      new("FLXcomponent", parameters = list(coef = coef),
          logLik = logLik, predict = predict, df = df)
    })
    z@fit <- function(x, y, w, component) {
      fit <- glm.fit(x, y, weights = w, family = poisson(),
                     offset = offset, start = component$coef)
      with(list(coef = coef(fit), df = ncol(x)), eval(z@defineComponent))
    }
  }
  else if (family == "Gamma") {
    z@defineComponent <- expression({
      predict <- function(x, ...) {
        dotarg = list(...)
        if ("offset" %in% names(dotarg)) offset <- dotarg$offset
        p <- x %*% coef
        if (!is.null(offset)) p <- p + offset
        get(family, mode = "function")()$linkinv(p)
      }
      logLik <- function(x, y, ...) dgamma(y, shape = shape,
                                           scale = predict(x, ...)/shape, log = TRUE)
      new("FLXcomponent", parameters = list(coef = coef,
                                            shape = shape), predict = predict, logLik = logLik,
          df = df)
    })
    z@fit <- function(x, y, w, component) {
      fit <- glm.fit(x, y, weights = w, family = Gamma(),
                     offset = offset, start = component$coef)
      with(list(coef = coef(fit), df = ncol(x) + 1, shape = sum(fit$prior.weights)/fit$deviance),
           eval(z@defineComponent))
    }
  }
  else stop(paste("Unknown family", family))
  z
}
