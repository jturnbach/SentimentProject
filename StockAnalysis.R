rm(list = ls())

library(quantmod)
library(forecast)
library(ggplot2)
library(readxl)
library(xgboost)
library(randomForest)
library(nnet)
library(e1071)


getSymbols("AAPL")
getSymbols("GOOG")
getSymbols("TSLA")
getSymbols("INTC")
getSymbols("TMUS")
getSymbols("AMZN")
getSymbols("DJI")

start = "2020-03-05"
end = "2020-04-26"
apple = window(AAPL$AAPL.Adjusted, start = start, end = end)
google = window(GOOG$GOOG.Adjusted, start = start, end = end)
tesla = window(TSLA$TSLA.Adjusted, start = start, end = end)
intel = window(INTC$INTC.Adjusted, start = start, end = end)
tmo = window(TMUS$TMUS.Adjusted, start = start, end = end)
amazon = window(AMZN$AMZN.Adjusted, start = start, end = end)
djia = window(DJI$DJI.Adjusted, start = start, end = end)

apple.pdiff = apple
for (i in 2:length(apple))
  apple.pdiff[i] = (as.numeric(apple[i]) - as.numeric(apple[i-1])) / as.numeric(apple[i-1])
apple.pdiff = window(apple.pdiff, start = time(apple.pdiff[2]))

google.pdiff = google
for (i in 2:length(google))
  google.pdiff[i] = (as.numeric(google[i]) - as.numeric(google[i-1])) / as.numeric(google[i-1])
google.pdiff = window(google.pdiff, start = time(google.pdiff[2]))

tesla.pdiff = tesla
for (i in 2:length(tesla))
  tesla.pdiff[i] = (as.numeric(tesla[i]) - as.numeric(tesla[i-1])) / as.numeric(tesla[i-1])
tesla.pdiff = window(tesla.pdiff, start = time(tesla.pdiff[2]))

intel.pdiff = intel
for (i in 2:length(intel))
  intel.pdiff[i] = (as.numeric(intel[i]) - as.numeric(intel[i-1])) / as.numeric(intel[i-1])
intel.pdiff = window(intel.pdiff, start = time(intel.pdiff[2]))

tmo.pdiff = tmo
for (i in 2:length(tmo))
  tmo.pdiff[i] = (as.numeric(tmo[i]) - as.numeric(tmo[i-1])) / as.numeric(tmo[i-1])
tmo.pdiff = window(tmo.pdiff, start = time(tmo.pdiff[2]))

amazon.pdiff = amazon
for (i in 2:length(amazon))
  amazon.pdiff[i] = (as.numeric(amazon[i]) - as.numeric(amazon[i-1])) / as.numeric(amazon[i-1])
amazon.pdiff = window(amazon.pdiff, start = time(amazon.pdiff[2]))

djia.pdiff = djia
for (i in 2:length(djia))
  djia.pdiff[i] = (as.numeric(djia[i]) - as.numeric(djia[i-1])) / as.numeric(djia[i-1])
djia.pdiff = window(djia.pdiff, start = time(djia.pdiff[2]))

apple.pdiff.corrected = apple.pdiff - djia.pdiff
google.pdiff.corrected = google.pdiff - djia.pdiff
tesla.pdiff.corrected = tesla.pdiff - djia.pdiff
intel.pdiff.corrected = intel.pdiff - djia.pdiff
tmo.pdiff.corrected = tmo.pdiff - djia.pdiff
amazon.pdiff.corrected = amazon.pdiff - djia.pdiff

dat = cbind(apple.pdiff, google.pdiff, tesla.pdiff, intel.pdiff, tmo.pdiff, amazon.pdiff, djia.pdiff)
colnames(dat) = c("Apple", "Google", "Tesla", "Intel", "T-Mobile", "Amazon", "DJIA")
autoplot(dat)
autoplot(cbind(apple.pdiff.corrected, google.pdiff.corrected, tesla.pdiff.corrected, intel.pdiff.corrected, tmo.pdiff.corrected, amazon.pdiff.corrected))

autoplot(tesla.pdiff)
autoplot(cbind(tesla.pdiff, djia.pdiff))
autoplot(tesla.pdiff.corrected)




#########################
# Import sentiment scores
#########################

tesla.sent = read_excel("SentimentScoresAverage.xlsx")
tesla.sent$Date = as.Date(tesla.sent$Date)
tesla.sent = xts(x = cbind(tesla.sent$Polarity, tesla.sent$Subjectivity, tesla.sent$`Aggregate Score`), order.by = tesla.sent$Date)
colnames(tesla.sent) = c("Polarity", "Subjectivity", "Aggregate.Score")

# Plot

autoplot(cbind(scale(tesla.pdiff.corrected), scale(tesla.sent)))

df = cbind(scale(tesla.pdiff.corrected), scale(tesla.sent$Aggregate.Score))
colnames(df) = c("Tesla Stock Movement", "Aggregate Score")
autoplot(df)
df = cbind(scale(tesla.pdiff.corrected), -scale(tesla.sent$Polarity))
colnames(df) = c("Tesla Stock Movement", "Polarity")
autoplot(df)
df = cbind(scale(tesla.pdiff.corrected), scale(tesla.sent$Subjectivity))
colnames(df) = c("Tesla Stock Movement", "Subjectivity")
autoplot(df)


# Fit models

acc = function(dat)
{
  sum = 0
  
  for (i in 1:length(dat[,1]))
  {
    if (dat$pred.move[i] == -1)
      if (dat$truth[i] <= -0.005)
        sum = sum + 1
    if (dat$pred.move[i] == 1)
      if (dat$truth[i] >= 0.005)
        sum = sum + 1
    if (dat$pred.move[i] == 0)
      if (-0.01 <= dat$truth[i] & dat$truth[i] <= 0.01)
        sum = sum + 1
  }
  
  return(sum / length(dat[,1]))
}

tesla.comb = cbind(tesla.pdiff.corrected, tesla.sent)
colnames(tesla.comb)[1] = "index"
cor(tesla.comb$index, tesla.comb$Polarity, use = "pairwise.complete.obs")
cor(tesla.comb$index, tesla.comb$Subjectivity, use = "pairwise.complete.obs")
cor(tesla.comb$index, tesla.comb$Aggregate.Score, use = "pairwise.complete.obs")
tesla.matrix = data.frame(tesla.comb)
colnames(tesla.matrix) = c("index", "Polarity", "Subjectivity", "Aggregate.Score")
days = nrow(tesla.matrix)
Polarity.1 = c(NA, as.vector(tesla.comb$Polarity[1:(days-1)]))
Polarity.2 = c(NA, NA, as.vector(tesla.comb$Polarity[1:(days-2)]))
Polarity.3 = c(NA, NA, NA, as.vector(tesla.comb$Polarity[1:(days-3)]))
Subjectivity.1 = c(NA, as.vector(tesla.comb$Subjectivity[1:(days-1)]))
Subjectivity.2 = c(NA, NA, as.vector(tesla.comb$Subjectivity[1:(days-2)]))
Subjectivity.3 = c(NA, NA, NA, as.vector(tesla.comb$Subjectivity[1:(days-3)]))
tesla.matrix = cbind(tesla.matrix, Polarity.1, Polarity.2, Polarity.3, Subjectivity.1, Subjectivity.2, Subjectivity.3)
tesla.train = window(xts(tesla.matrix, order.by = as.Date(rownames(tesla.matrix))), end = '2020-4-10')
tesla.test = window(xts(tesla.matrix, order.by = as.Date(rownames(tesla.matrix))), start = '2020-4-11')










tesla.lm = lm(index ~ Polarity + Subjectivity, data = tesla.matrix)
summary(tesla.lm)
tesla.lm = lm(index ~ Polarity.1 + Polarity.2 + Polarity.3 + Subjectivity.1 + Subjectivity.2 + Subjectivity.3, data = tesla.train)
summary(tesla.lm)
lmdf = cbind(xts(predict(tesla.lm, newdata = tesla.train), order.by = time(tesla.train)), window(tesla.pdiff.corrected, end = "2020-4-10"))
colnames(lmdf) = c("Predicted Stock Movement", "Actual Stock Movement")
autoplot(lmdf) + ggtitle("Training Data")
lmpred = predict(tesla.lm, newdata = tesla.test)
lmdf = cbind(xts(predict(tesla.lm, newdata = tesla.test), order.by = time(tesla.test)), window(tesla.pdiff.corrected, start = "2020-4-11"))
colnames(lmdf) = c("Predicted Stock Movement", "Actual Stock Movement")
autoplot(lmdf) + ggtitle("Test Data")
lmpred.move = ifelse(lmpred >= 0.005, yes = 1, no = ifelse(lmpred <= -0.005, yes = -1, no = 0))
lmpred.move = xts(lmpred.move, order.by = as.Date(names(lmpred.move)))
test = cbind(lmpred.move, window(tesla.pdiff.corrected, start = "2020-4-11"))
test = na.omit(test)
colnames(test) = c("pred.move", "truth")
print(acc(test))
lmpred = predict(tesla.lm, newdata = tesla.matrix)
lmdf = cbind(xts(lmpred, order.by = as.Date(rownames(tesla.matrix))), tesla.pdiff.corrected)
colnames(lmdf) = c("Predicted Stock Movement", "Actual Stock Movement")
autoplot(lmdf) + ggtitle("All Data")
lmpred.move = ifelse(lmpred >= 0.005, yes = 1, no = ifelse(lmpred <= -0.005, yes = -1, no = 0))
lmpred.move = xts(lmpred.move, order.by = as.Date(names(lmpred.move)))
test = cbind(lmpred.move, tesla.pdiff.corrected)
test = na.omit(test)
colnames(test) = c("pred.move", "truth")
lm_acc = acc(test)








tesla.matrix = na.omit(tesla.matrix)
tesla.train = na.omit(tesla.train)
best.mtry.results <- tuneRF(tesla.train[, -ncol(tesla.train)], tesla.train$index, 
                            stepFactor=1.5, improve=0.01, trace=TRUE)

# Display the matrix to understand its structure
print(best.mtry.results)

# Typically, the first column is the OOB error, and row names are the mtry values:
# Find the row with the minimum OOB error
best.mtry.index <- which.min(best.mtry.results[, 1])  # assuming column 1 holds OOB errors
best.mtry.value <- as.numeric(rownames(best.mtry.results)[best.mtry.index])
print(best.mtry.value)
tesla.randomForest = randomForest(index ~ Polarity + Subjectivity, data = tesla.matrix, mtry = best.mtry.value)
summary(tesla.randomForest)
tesla.randomForest = randomForest(index ~ Polarity.1 + Polarity.2 + Polarity.3 + Subjectivity.1 + Subjectivity.2 + Subjectivity.3, data = tesla.train, mtry = best.mtry.value)
summary(tesla.randomForest)

rfdf = cbind(xts(predict(tesla.randomForest, newdata = tesla.train), order.by = time(tesla.train)), window(tesla.pdiff.corrected, end = "2020-4-10"))
colnames(rfdf) = c("Predicted Stock Movement", "Actual Stock Movement")
autoplot(rfdf) + ggtitle("Training Data")
rfpred = predict(tesla.randomForest, newdata = tesla.test)
rfdf = cbind(xts(predict(tesla.randomForest, newdata = tesla.test), order.by = time(tesla.test)), window(tesla.pdiff.corrected, start = "2020-4-11"))
colnames(rfdf) = c("Predicted Stock Movement", "Actual Stock Movement")
autoplot(rfdf) + ggtitle("Test Data")
rfpred.move = ifelse(rfpred >= 0.005, yes = 1, no = ifelse(rfpred <= -0.005, yes = -1, no = 0))
rfpred.move = xts(rfpred.move, order.by = as.Date(names(rfpred.move)))
test = cbind(rfpred.move, window(tesla.pdiff.corrected, start = "2020-4-11"))
test = na.omit(test)
colnames(test) = c("pred.move", "truth")
print(acc(test))
rfpred = predict(tesla.randomForest, newdata = tesla.matrix)
rfdf = cbind(xts(rfpred, order.by = as.Date(rownames(tesla.matrix))), tesla.pdiff.corrected)
colnames(df) = c("Predicted Stock Movement", "Actual Stock Movement")
autoplot(df) + ggtitle("All Data")
rfpred.move = ifelse(rfpred >= 0.005, yes = 1, no = ifelse(rfpred <= -0.005, yes = -1, no = 0))
rfpred.move = xts(rfpred.move, order.by = as.Date(names(rfpred.move)))
test = cbind(rfpred.move, tesla.pdiff.corrected)
test = na.omit(test)
colnames(test) = c("pred.move", "truth")
rf_acc = acc(test)

results_df <- data.frame(
  model = c("Linear Regression", "RandomForest"),
  accuracy = c(lm_acc, rf_acc)
)

results_df




dtrain <- xgb.DMatrix(data = as.matrix(tesla.train[, -1]), label = tesla.train$index)
dtest <- xgb.DMatrix(data = as.matrix(tesla.test[, -1]), label = tesla.test$index)
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.5,
  colsample_bytree = 0.5
)
nrounds <- 100
tesla.xgb <- xgb.train(params = params, data = dtrain, nrounds = nrounds)
predictions.train <- predict(tesla.xgb, dtrain)
predictions.test <- predict(tesla.xgb, dtest)
xts_train <- xts(predictions.train, order.by = as.Date(rownames(tesla.train)))
xts_test <- xts(predictions.test, order.by = as.Date(rownames(tesla.test)))
df_train <- cbind(xts_train, window(tesla.pdiff.corrected, end = "2020-4-10"))
df_test <- cbind(xts_test, window(tesla.pdiff.corrected, start = "2020-4-11"))
colnames(test) = c("pred.move", "truth")
print(acc(test))










tesla.glm = glm(index ~ Polarity + Subjectivity, data = tesla.matrix)
summary(tesla.glm)
tesla.glm = glm(index ~ Polarity.1 + Polarity.2 + Polarity.3 + Subjectivity.1 + Subjectivity.2 + Subjectivity.3, data = tesla.train)
summary(tesla.glm)
df = cbind(xts(predict(tesla.glm, newdata = tesla.train), order.by = time(tesla.train)), window(tesla.pdiff.corrected, end = "2020-4-10"))
colnames(df) = c("Predicted Stock Movement", "Actual Stock Movement")
autoplot(df) + ggtitle("Training Data")
pred = predict(tesla.glm, newdata = tesla.test)
df = cbind(xts(predict(tesla.glm, newdata = tesla.test), order.by = time(tesla.test)), window(tesla.pdiff.corrected, start = "2020-4-11"))
colnames(df) = c("Predicted Stock Movement", "Actual Stock Movement")
autoplot(df) + ggtitle("Test Data")
pred.move = ifelse(pred >= 0.005, yes = 1, no = ifelse(pred <= -0.005, yes = -1, no = 0))
pred.move = xts(pred.move, order.by = as.Date(names(pred.move)))
test = cbind(pred.move, window(tesla.pdiff.corrected, start = "2020-4-11"))
test = na.omit(test)
colnames(test) = c("pred.move", "truth")
print(acc(test))
pred = predict(tesla.glm, newdata = tesla.matrix)
df = cbind(xts(pred, order.by = as.Date(rownames(tesla.matrix))), tesla.pdiff.corrected)
colnames(df) = c("Predicted Stock Movement", "Actual Stock Movement")
autoplot(df) + ggtitle("All Data")
pred.move = ifelse(pred >= 0.005, yes = 1, no = ifelse(pred <= -0.005, yes = -1, no = 0))
pred.move = xts(pred.move, order.by = as.Date(names(pred.move)))
test = cbind(pred.move, tesla.pdiff.corrected)
test = na.omit(test)
colnames(test) = c("pred.move", "truth")
print(acc(test))










tesla.nn = nnet(index ~ Polarity + Subjectivity, data = tesla.matrix, size = 10)
summary(tesla.nn)
tesla.nn = nnet(index ~ Polarity.1 + Polarity.2 + Polarity.3 + Subjectivity.1 + Subjectivity.2 + Subjectivity.3, data = tesla.train, size = 10)
summary(tesla.nn)
df = cbind(xts(predict(tesla.nn, newdata = tesla.train), order.by = time(tesla.train)), window(tesla.pdiff.corrected, end = "2020-4-10"))
colnames(df) = c("Predicted Stock Movement", "Actual Stock Movement")
autoplot(df) + ggtitle("Training Data")
pred = predict(tesla.nn, newdata = tesla.test)
df = cbind(xts(predict(tesla.nn, newdata = tesla.test), order.by = time(tesla.test)), window(tesla.pdiff.corrected, start = "2020-4-11"))
colnames(df) = c("Predicted Stock Movement", "Actual Stock Movement")
autoplot(df) + ggtitle("Test Data")
pred.move = ifelse(pred >= 0.005, yes = 1, no = ifelse(pred <= -0.005, yes = -1, no = 0))
pred.move = xts(pred.move, order.by = as.Date(names(pred.move)))
test = cbind(pred.move, window(tesla.pdiff.corrected, start = "2020-4-11"))
test = na.omit(test)
colnames(test) = c("pred.move", "truth")
print(acc(test))
pred = predict(tesla.nn, newdata = tesla.matrix)
df = cbind(xts(pred, order.by = as.Date(rownames(tesla.matrix))), tesla.pdiff.corrected)
colnames(df) = c("Predicted Stock Movement", "Actual Stock Movement")
autoplot(df) + ggtitle("All Data")
pred.move = ifelse(pred >= 0.005, yes = 1, no = ifelse(pred <= -0.005, yes = -1, no = 0))
pred.move = xts(pred.move, order.by = as.Date(names(pred.move)))
test = cbind(pred.move, tesla.pdiff.corrected)
test = na.omit(test)
colnames(test) = c("pred.move", "truth")
print(acc(test))

# Trying with a sliding window (swa = sliding window average)

for (window in 1 + 2*(0:4))
{
  tesla.comb.swa = tesla.comb
  for (i in floor(window/2):(length(tesla.comb.swa$Polarity)-floor(window/2)))
  {
    tesla.comb.swa$Polarity[i] = mean(tesla.comb$Polarity[(i-floor(window/2)):(i+floor(window/2))])
    tesla.comb.swa$Subjectivity[i] = mean(tesla.comb$Subjectivity[(i-floor(window/2)):(i+floor(window/2))])
    tesla.comb.swa$Aggregate.Score[i] = mean(tesla.comb$Aggregate.Score[(i-floor(window/2)):(i+floor(window/2))])
  }
  print(cor(tesla.comb.swa$index, tesla.comb.swa$Polarity, use = "pairwise.complete.obs"))
  print(cor(tesla.comb.swa$index, tesla.comb.swa$Subjectivity, use = "pairwise.complete.obs"))
  print(cor(tesla.comb.swa$index, tesla.comb.swa$Aggregate.Score, use = "pairwise.complete.obs"))
}

# Try with uncorrected stock movement

tesla.comb = cbind(tesla.pdiff, tesla.sent)
colnames(tesla.comb)[1] = "index"
window = 3
tesla.comb.swa = tesla.comb
for (i in floor(window/2):(length(tesla.comb.swa$Polarity)-floor(window/2)))
{
  tesla.comb.swa$Polarity[i] = mean(tesla.comb$Polarity[(i-floor(window/2)):(i+floor(window/2))])
  tesla.comb.swa$Subjectivity[i] = mean(tesla.comb$Subjectivity[(i-floor(window/2)):(i+floor(window/2))])
  tesla.comb.swa$Aggregate.Score[i] = mean(tesla.comb$Aggregate.Score[(i-floor(window/2)):(i+floor(window/2))])
}
print(cor(tesla.comb.swa$index, tesla.comb.swa$Polarity, use = "pairwise.complete.obs"))
print(cor(tesla.comb.swa$index, tesla.comb.swa$Subjectivity, use = "pairwise.complete.obs"))
print(cor(tesla.comb.swa$index, tesla.comb.swa$Aggregate.Score, use = "pairwise.complete.obs"))


# Plot

  # Plotting negative because the correlation is negative
df = cbind(scale(tesla.pdiff.corrected), -scale(tesla.comb.swa$Aggregate.Score))
colnames(df) = c("Tesla Stock Movement", "-Aggregate Score")
autoplot(df)
df = cbind(scale(tesla.pdiff.corrected), -scale(tesla.comb.swa$Polarity))
colnames(df) = c("Tesla Stock Movement", "-Polarity")
autoplot(df)
df = cbind(scale(tesla.pdiff.corrected), -scale(tesla.comb.swa$Subjectivity))
colnames(df) = c("Tesla Stock Movement", "-Subjectivity")
autoplot(df)

df = cbind(scale(tesla.pdiff.corrected), -scale(tesla.comb.swa$Aggregate.Score))
colnames(df) = c("Tesla Stock Movement", "-Aggregate Score")
ggplot(df, aes(x = seq_along(`Tesla Stock Movement`))) + 
  geom_line(aes(y = `Tesla Stock Movement`, color = "Tesla Stock Movement"), na.rm=TRUE) + 
  geom_line(aes(y = `-Aggregate Score`, color = "-Aggregate Score"), na.rm=TRUE) +
  labs(y = "Scaled Values", x = "Time/Index") +
  scale_colour_manual(values = c("Tesla Stock Movement" = "blue", "-Aggregate Score" = "red")) +
  theme_minimal() +
  ggtitle("Comparison of Tesla Stock Movement and -Aggregate Score")



df = cbind(scale(tesla.pdiff.corrected), -scale(tesla.comb.swa$Polarity))
colnames(df) = c("Tesla Stock Movement", "-Polarity")
ggplot(df, aes(x = seq_along(`Tesla Stock Movement`))) + 
  geom_line(aes(y = `Tesla Stock Movement`, color = "Tesla Stock Movement"), na.rm=TRUE) + 
  geom_line(aes(y = `-Polarity`, color = "-Polarity"), na.rm=TRUE) +
  labs(y = "Scaled Values", x = "Time/Index") +
  scale_colour_manual(values = c("Tesla Stock Movement" = "blue", "-Polarity" = "red")) +
  theme_minimal() +
  ggtitle("Comparison of Tesla Stock Movement and -Polarity")


df = cbind(scale(tesla.pdiff.corrected), -scale(tesla.comb.swa$Subjectivity))
colnames(df) = c("Tesla Stock Movement", "-Subjectivity")
ggplot(df, aes(x = seq_along(`Tesla Stock Movement`))) + 
  geom_line(aes(y = `Tesla Stock Movement`, color = "Tesla Stock Movement"), na.rm=TRUE) + 
  geom_line(aes(y = `-Subjectivity`, color = "-Subjectivity"), na.rm=TRUE) +
  labs(y = "Scaled Values", x = "Time/Index") +
  scale_colour_manual(values = c("Tesla Stock Movement" = "blue", "-Subjectivity" = "red")) +
  theme_minimal() +
  ggtitle("Comparison of Tesla Stock Movement and -Subjectivity")

# Output predictions to file

pred = predict(tesla.lm, newdata = tesla.matrix)
dat = cbind(pred, ifelse(pred >= 0.005, yes = 1, no = ifelse(pred <= -0.005, yes = -1, no = 0)))
dat = na.omit(dat)
write.table(x = dat, sep = " ", file = "Tesla-predict.txt")













