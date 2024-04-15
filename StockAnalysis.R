rm(list = ls())

library(quantmod)
library(forecast)
library(ggplot2)
library(readxl)
library(randomForest)
library(mice)
library(caret)

getSymbols("TSLA")
getSymbols("DJI")

start = "2020-03-05"
end = "2020-04-26"
tesla = window(TSLA$TSLA.Adjusted, start = start, end = end)
djia = window(DJI$DJI.Adjusted, start = start, end = end)

tesla.pdiff = tesla
for (i in 2:length(tesla))
  tesla.pdiff[i] = (as.numeric(tesla[i]) - as.numeric(tesla[i-1])) / as.numeric(tesla[i-1])
tesla.pdiff = window(tesla.pdiff, start = time(tesla.pdiff[2]))


djia.pdiff = djia
for (i in 2:length(djia))
  djia.pdiff[i] = (as.numeric(djia[i]) - as.numeric(djia[i-1])) / as.numeric(djia[i-1])
djia.pdiff = window(djia.pdiff, start = time(djia.pdiff[2]))

tesla.pdiff.corrected = tesla.pdiff - djia.pdiff

dat = cbind(tesla.pdiff, djia.pdiff)
colnames(dat) = c("Tesla", "DJIA")
autoplot(dat)
autoplot(cbind(tesla.pdiff.corrected))

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




#########################
# Fit models
#########################



acc <- function(dat, model) {
  pred_col_name <- paste0(model, "pred.move")
  
  correct <- with(dat, {
    (get(pred_col_name) == -1 & truth <= -0.005) |
      (get(pred_col_name) == 1 & truth >= 0.005) |
      (get(pred_col_name) == 0 & truth >= -0.01 & truth <= 0.01)
  })
  
  mean(correct)
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
tesla.train_df <- data.frame(date = index(tesla.train), coredata(tesla.train))
tesla.test_df <- data.frame(date = index(tesla.test), coredata(tesla.test))

tesla.matrix.df <- as.data.frame(tesla.matrix)
imputed_data <- mice(tesla.matrix.df, method='rf', m=5, seed=500)
completed_data <- complete(imputed_data)

tesla.matrix.imputed <- xts(completed_data, order.by = as.Date(rownames(completed_data)))

tesla.train.imputed <- window(tesla.matrix.imputed, end = '2020-4-10')
tesla.test.imputed <- window(tesla.matrix.imputed, start = '2020-4-11')






# LM Model
#########################

tesla.lm = lm(index ~ Polarity.1 + Polarity.2 + Polarity.3 + Subjectivity.1 + Subjectivity.2 + Subjectivity.3, data = tesla.train)
lmdf = cbind(xts(predict(tesla.lm, newdata = tesla.train), order.by = time(tesla.train)), window(tesla.pdiff.corrected, end = "2020-4-10"))
lmpred = predict(tesla.lm, newdata = tesla.test)
lmdf = cbind(xts(predict(tesla.lm, newdata = tesla.test), order.by = time(tesla.test)), window(tesla.pdiff.corrected, start = "2020-4-11"))
lmpred.move = ifelse(lmpred >= 0.005, yes = 1, no = ifelse(lmpred <= -0.005, yes = -1, no = 0))
lmpred.move = xts(lmpred.move, order.by = as.Date(names(lmpred.move)))
lmtest = cbind(lmpred.move, window(tesla.pdiff.corrected, start = "2020-4-11"))
lmtest = na.omit(lmtest)
colnames(lmtest) = c("pred.move", "truth")
lmpred = predict(tesla.lm, newdata = tesla.matrix)
lmdf = cbind(xts(lmpred, order.by = as.Date(rownames(tesla.matrix))), tesla.pdiff.corrected)
colnames(lmdf) = c("Predicted Stock Movement", "Actual Stock Movement")
lmpred.move = ifelse(lmpred >= 0.005, yes = 1, no = ifelse(lmpred <= -0.005, yes = -1, no = 0))
lmpred.move = xts(lmpred.move, order.by = as.Date(names(lmpred.move)))
lmtest = cbind(lmpred.move, tesla.pdiff.corrected)
lmtest = na.omit(lmtest)
colnames(lmtest) = c("pred.move", "truth")
lm_acc = acc(lmtest, "lm")
lmpred = xts(lmpred, order.by = as.Date(names(lmpred)))
autoplot(lmdf)
print(lm_acc)




# RandomForest Model
#########################

set.seed(123)
fitControl <- trainControl(
  method = "cv",
  number = 10
)

modelTuning <- train(
  index ~ Polarity.1 + Polarity.2 + Polarity.3 + Subjectivity.1 + Subjectivity.2 + Subjectivity.3,
  data = tesla.train.imputed,
  method = "rf",
  trControl = fitControl,
  tuneGrid = expand.grid(mtry = 1:20),
  ntree = 500
)

bestMtry <- modelTuning$bestTune$mtry

set.seed(123)
tesla.randomForest = randomForest(index ~ Polarity.1 + Polarity.2 + Polarity.3 + Subjectivity.1 + Subjectivity.2 + Subjectivity.3, data = tesla.train.imputed, mtry = bestMtry, ntree = 500)
rfdf = cbind(xts(predict(tesla.randomForest, newdata = tesla.train.imputed), order.by = time(tesla.train)), window(tesla.pdiff.corrected, end = "2020-4-10"))
rfpred = predict(tesla.randomForest, newdata = tesla.test.imputed)
rfdf = cbind(xts(predict(tesla.randomForest, newdata = tesla.test.imputed), order.by = time(tesla.test)), window(tesla.pdiff.corrected, start = "2020-4-11"))
rfpred.move = ifelse(rfpred >= 0.005, yes = 1, no = ifelse(rfpred <= -0.005, yes = -1, no = 0))
rfpred.move = xts(rfpred.move, order.by = as.Date(names(rfpred.move)))
rftest = cbind(rfpred.move, window(tesla.pdiff.corrected, start = "2020-4-11"))
rftest = na.omit(rftest)
colnames(rftest) = c("pred.move", "truth")
rfpred = predict(tesla.randomForest, newdata = tesla.matrix.imputed)
rfdf = cbind(xts(rfpred, order.by = as.Date(rownames(tesla.matrix))), tesla.pdiff.corrected)
colnames(rfdf) = c("Predicted Stock Movement", "Actual Stock Movement")
rfpred.move = ifelse(rfpred >= 0.005, yes = 1, no = ifelse(rfpred <= -0.005, yes = -1, no = 0))
rfpred.move = xts(rfpred.move, order.by = as.Date(names(rfpred.move)))
rftest = cbind(rfpred.move, tesla.pdiff.corrected)
rftest = na.omit(rftest)
colnames(rftest) = c("pred.move", "truth")
rf_acc = acc(rftest, "rf")
rfpred = xts(rfpred, order.by = as.Date(names(rfpred)))
autoplot(rfdf)
print(rf_acc)



#########################
# Comparisons
#########################


results_df = data.frame(
  model = c("Linear Regression", "RandomForest"),
  accuracy = c(lm_acc, rf_acc)
)

results_df


df = cbind(scale(tesla.pdiff.corrected), scale(lmpred), scale(rfpred))
colnames(df) = c("Tesla Stock Movement", "Linear Regression Predictions", "RandomForest Predictions")
ggplot(df, aes(x = seq_along(`Tesla Stock Movement`))) + 
  geom_line(aes(y = `Tesla Stock Movement`, color = "Tesla Stock Movement"), na.rm=TRUE) + 
  geom_line(aes(y = `Linear Regression Predictions`, color = "Linear Regression Predictions"), na.rm=TRUE) +
  geom_line(aes(y = `RandomForest Predictions`, color = "RandomForest Predictions"), na.rm=TRUE) +
  labs(y = "Scaled Values", x = "Time/Index") +
  scale_colour_manual(values = c("Tesla Stock Movement" = "black", "Linear Regression Predictions" = "blue", "RandomForest Predictions" = "red")) +
  ggtitle("Comparison of Tesla Stock Movement Predictions of LM and RF Models")

df = cbind(scale(tesla.pdiff.corrected), scale(lmpred))
colnames(df) = c("Tesla Stock Movement", "Linear Regression Predictions")
ggplot(df, aes(x = seq_along(`Tesla Stock Movement`))) + 
  geom_line(aes(y = `Tesla Stock Movement`, color = "Tesla Stock Movement"), na.rm=TRUE) + 
  geom_line(aes(y = `Linear Regression Predictions`, color = "Linear Regression Predictions"), na.rm=TRUE) +
  labs(y = "Scaled Values", x = "Time/Index") +
  scale_colour_manual(values = c("Tesla Stock Movement" = "black", "Linear Regression Predictions" = "blue")) +
  ggtitle("Comparison of Tesla Stock Movement Predictions of LM Model")

df = cbind(scale(tesla.pdiff.corrected), scale(rfpred))
colnames(df) = c("Tesla Stock Movement", "RandomForest Predictions")
ggplot(df, aes(x = seq_along(`Tesla Stock Movement`))) + 
  geom_line(aes(y = `Tesla Stock Movement`, color = "Tesla Stock Movement"), na.rm=TRUE) + 
  geom_line(aes(y = `RandomForest Predictions`, color = "RandomForest Predictions"), na.rm=TRUE) +
  labs(y = "Scaled Values", x = "Time/Index") +
  scale_colour_manual(values = c("Tesla Stock Movement" = "black", "RandomForest Predictions" = "red")) +
  ggtitle("Comparison of Tesla Stock Movement Predictions of RF Model")

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

lmpred = predict(tesla.lm, newdata = tesla.matrix)
rfpred = predict(tesla.randomForest, newdata = tesla.matrix.imputed)
lmdat = cbind(lmpred, ifelse(lmpred >= 0.005, yes = 1, no = ifelse(lmpred <= -0.005, yes = -1, no = 0)))
rfdat = cbind(rfpred, ifelse(rfpred >= 0.005, yes = 1, no = ifelse(rfpred <= -0.005, yes = -1, no = 0)))
joined_data <- merge(lmdat, rfdat, by = "row.names", all = TRUE)
write.table(x = joined_data, sep = " ", file = "Tesla-predict.txt")













