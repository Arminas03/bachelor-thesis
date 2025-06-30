# =====================
# LIBRARIES
# =====================

library(highfrequency)
library(xts)
library(zoo)
library(MASS)
library(arfima)

# =====================
# DATA
# =====================
data_jae <- read.csv(file.choose())

names(data_jae)[1] <- "Date"
data_jae$Date <- as.Date(data_jae$Date, format = "%m/%d/%y")

# =====================
# ARFIMA
# =====================

run_arfima_regression <- function(regressor, arfima_order, horizon) {
  n <- length(regressor)
  split_index = 1000
  n_test <- n - split_index - horizon + 1
  
  pred_test <- regressor[(split_index + 1):n]
  pred_test <- rollapply(pred_test, horizon, sum, align="left", fill = NA)
  pred_test <- pred_test[!is.na(pred_test)]
  arfima_pred <- numeric(n_test)
  
  prev_time <- Sys.time()
  for (i in 1:n_test) {
    curr_window <- regressor[i:(split_index + i - 1)]
    
    invisible(capture.output({
      model <- arfima(curr_window, order = arfima_order)
    }))
    y_pred <- sum(predict(model, n.ahead = horizon)[[1]]$Forecast)
    
    y_min <- min(curr_window)
    y_max <- max(curr_window)
    if (y_pred > y_max) y_pred <- y_max
    if (y_pred < y_min) y_pred <- y_min
    arfima_pred[i] <- y_pred
    
    if (i %% 500 == 0) {
      time_elapsed <- round(as.numeric(Sys.time() - prev_time, units = "secs"), 2)
      cat(i, "/", n_test, ", Time: ", time_elapsed, "s\n")
      prev_time <- Sys.time()
    }
  }
  
  se <- (pred_test - arfima_pred)^2
  qlike <- (pred_test / arfima_pred) - log(pred_test / arfima_pred) - 1
  
  return(list(
    mean_se = mean(se), median_se = median(se),
    mean_qlike = mean(qlike), median_qlike = median(qlike)
  ))
}

results_arfima <- data.frame(
  regressor = character(),
  order_p = integer(),
  order_d = integer(),
  order_q = integer(),
  horizon = integer(),
  mean_se = numeric(),
  median_se = numeric(),
  mean_qlike = numeric(),
  median_qlike = numeric(),
  stringsAsFactors = FALSE
)
regressors <- list(RV = data_jae$RV, ORV = data_jae$ORV)

for (regressor_name in names(regressors)) {
  regressor <- regressors[[regressor_name]]
  for (arfima_order in list(c(1,0,1), c(5,0,0))) {
    for (horizon in list(1,5,22)) {
      cat(
        "Running ", regressor_name, "order = ",
        arfima_order, " horizon = ", horizon, "\n"
      )
      if (regressor_name != "ORV" || horizon < 20 || arfima_order[1] != 5) {
        next
      }
      res <- run_arfima_regression(regressor, arfima_order, horizon)
      
      results_arfima <- rbind(results_arfima, data.frame(
        regressor = regressor_name,
        order_p = arfima_order[1],
        order_d = arfima_order[2],
        order_q = arfima_order[3],
        horizon = horizon,
        mean_se = res$mean_se,
        median_se = res$median_se,
        mean_qlike = res$mean_qlike,
        median_qlike = res$median_qlike,
        stringsAsFactors = FALSE
      ))
    }
  }
}


