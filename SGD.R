library(readxl)
library(dplyr)
library(ROCR)
library(sigmoid)

GD.lm <- function(formula, data, learning_rate = 0.001, err = 1e-02, 
                  epoch_num = 50000){
  Y <- model.response(model.frame(formula, data))
  X <- model.matrix(formula, data)
  p <- ncol(X)
  n <- nrow(X)
  
  w <- numeric(p)
  t <- 0
  
  repeat{
    t <- t + 1
    w <- w - learning_rate * 2 / n * t(X) %*% (X %*% w - Y)
    
    if (t >= epoch_num)
      break
  }
  
  R2 <- 1 - 1 / n * sum((X %*% w - Y) ^ 2) / var(Y)
  Adj_R2 <- 1 - ((1 - R2) * (n - 1) / (n - p + 1))
  return(list(coeffs = w, R2 = R2, Adj_R2 = Adj_R2))
}

Adam.lm <- function(formula, data, learning_rate = 0.001,
                       beta2 = 0.999, beta1 = 0.9, epsilon = 1e-08, 
                    err = 1e-08, AMSGrad = FALSE, batch_num = 1,
                    epoch_num = 50000){
  Y <- model.response(model.frame(formula, data))
  X <- model.matrix(formula, data)
  p <- ncol(X)
  n <- nrow(X)
 
  w.next <- numeric(p)
  setNames(w.next, colnames(X))
  t <- 0
  m.bias <- numeric(p)
  v.bias <- numeric(p)
  v.bias.max <- v.bias
  
  repeat{
    t <- t + 1
    w.prev <- w.next
    i <- sample(c(1:n), batch_num) 
    if (batch_num > 1){
      g <- 2 / batch_num * t(X[i,]) %*% (X[i,] %*% w.prev - Y[i]) # seems OK  
    }
    else{
      g <- 2 / batch_num * X[i,] %*% (X[i,] %*% w.prev - Y[i])
    }
    m.bias <- beta1 * m.bias + (1 - beta1) * g
    v.bias <- beta2 * v.bias + (1 - beta2) * g ^ 2
    m <- m.bias / (1 - beta1 ^ t)
    if (AMSGrad){
      v.bias.max <- pmax(v.bias, v.bias.max)
      v <- v.bias.max / (1 - beta2 ^ t)
    }
    else{
      v <- v.bias / (1 - beta2 ^ t)
    }
    w.next <- w.prev - learning_rate * m / (sqrt(v) + epsilon) 
    
    dist.next <- sqrt(sum((w.next - w.prev)^2))
    
    if (t > epoch_num){
      w <- w.next
      break
    }
    dist <- dist.next
  }
  dimnames(w) <- list(colnames(X), NULL)
  R2 <- 1 - 1 / n * sum((X %*% w.next - Y) ^ 2) / var(Y)
  Adj_R2 <- 1 - ((1 - R2) * (n - 1) / (n - p + 1))
  return(list(coeffs = w, R2 = R2, Adj_R2 = Adj_R2))
}

Adam.logreg <- function(formula, data, learning_rate = 0.001, 
                        beta2 = 0.999, beta1 = 0.9, epsilon = 1e-08,
                        err = 1e-08, AMSGrad = FALSE,
                        epoch_num = 50000){
  Y <- model.response(model.frame(formula, data))
  X <- model.matrix(formula, data)
  p <- ncol(X)
  n <- nrow(X)
  
  w.next <- numeric(p)
  setNames(w.next, colnames(X))
  t <- 0
  m.bias <- numeric(p)
  v.bias <- numeric(p)
  v.bias.max <- v.bias
  
  repeat{
    t <- t + 1
    w.prev <- w.next
    i <- sample(c(1:n), 1) 
    #g <- 2 / batch_num * X[i,] %*% (X[i,] %*% w.prev - Y[i])
    g <- -X[i, ] %*% (Y[i] - sigmoid(X[i, ] %*% w.prev)) 
    m.bias <- beta1 * m.bias + (1 - beta1) * g
    v.bias <- beta2 * v.bias + (1 - beta2) * g ^ 2
    m <- m.bias / (1 - beta1 ^ t)
    if (AMSGrad){
      v.bias.max <- pmax(v.bias, v.bias.max)
      v <- v.bias.max / (1 - beta2 ^ t)
    }
    else{
      v <- v.bias / (1 - beta2 ^ t)
    }
    w.next <- w.prev - learning_rate * m / (sqrt(v) + epsilon) 
    
    dist.next <- sqrt(sum((w.next - w.prev)^2))
    
    if (t > epoch_num){
      w <- w.next
      break
    }
    dist <- dist.next
  }
  dimnames(w) <- list(colnames(X), NULL)
  probs <- sigmoid(X %*% w)
  return(list(coeffs = w, probs = probs))
}

df <- read_excel("CARDATA.xls", )
df <- df %>% dplyr::select(-ROW) %>% arrange(MAKE, MODEL, YEAR, PRICE)
df$ORIGIN <- as.factor(df$ORIGIN)
df$YEAR <- as.factor(df$YEAR)

df.log <- df %>% mutate(DISPLACE = log(DISPLACE), 
                 HORSEPOW = log(HORSEPOW),
                 WEIGHT = log(WEIGHT),
                 PRICE = log(PRICE),
                 ACCEL = log(ACCEL))

df.log.out <- df.log %>% filter(CYLINDER %% 2 == 0 & PRICE <= 9.5)

formula <- scale(PRICE) ~ scale(WEIGHT) + scale(ACCEL) + scale(HORSEPOW) + 
  scale(DISPLACE) + scale(MPG) + scale(CYLINDER) + YEAR + ORIGIN

#Linear regression

model1 <- lm(formula, df.log.out)
summary(model1)
GD.result <- GD.lm(formula, df.log.out)
Adam.result <- Adam.lm(formula, df.log.out)

#Logistic regression

#ORIGIN == 1 - made in America, 0 - elsewhere
df.log.out.bin <- df.log.out %>% mutate(ORIGIN = ifelse(ORIGIN == 1, 1, 0))
df.log.out.bin <- na.omit(df.log.out.bin)
formula.classification <- ORIGIN ~ scale(PRICE) + scale(ACCEL) + 
  scale(HORSEPOW) + scale(DISPLACE) + scale(MPG) + scale(CYLINDER) + YEAR

logistic_model <- glm(formula.classification, data = df.log.out.bin,
                      family = "binomial")
summary(logistic_model)

Adam.logreg.result <- Adam.logreg(formula.classification, data = df.log.out.bin)

logistic_model.pred <- predict(logistic_model, newdata = df.log.out.bin, 
                                type = "response")
logistic_model.actual <- df.log.out.bin[["ORIGIN"]]
pred <- prediction(logistic_model.pred, logistic_model.actual)
perf <- performance(pred, "tpr", "fpr")
AUC <- performance(pred, measure = "auc")@y.values[[1]]

pred.Adam <- prediction(Adam.logreg.result$probs, logistic_model.actual)
perf.Adam <- performance(pred.Adam, "tpr", "fpr")
AUC.Adam <- performance(pred.Adam, measure = "auc")@y.values[[1]]

svg(filename = "ROC.svg")
plot(perf, col = "darkgreen")
plot(perf.Adam, col = "blue", add = TRUE)
abline(a = 0, b = 1, col = "red")
legend("bottomright", legend = c(sprintf("glm_ROC  AUC = %.3f", AUC), 
                                 sprintf("Adam_ROC  AUC = %.3f", AUC.Adam)),
       col = c("darkgreen", "blue"), lwd = 2)
dev.off()