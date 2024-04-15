# Sentiment-Driven Forecasts: Analyzing Social Media's Influence on Stock Market Trends

## Introduction
This project examines the influence of social media sentiment on stock prices, using Twitter data to predict market trends through sentiment analysis and machine learning models. The designed framework works in three parts:
 - Tweet Collection: Tweets relevant to a specified company are fetched using the Twitter API, based on simple search criteria. These tweets are then organized by their posting dates.
   
 - Sentiment Analysis: Utilizing the Taureau framework, which integrates the TextBlob sentiment analysis engine, tweets undergo a thorough sentiment extraction. The TextBlob engine combines various statistical methods, including Word2vec for text transformation and an ensemble of Naive Bayes and Random Forest classifiers for opinion mining. Each tweet is analyzed to yield two key metrics: Polarity and Subjectivity. Polarity measures the emotional orientation of the tweet, categorized as positive (close to 1), negative (close to -1), or neutral (around 0). Subjectivity assesses whether the content represents factual information (score close to 0) or personal opinions and feelings about the company (score close to 1).
   
 - Stock Movement Analysis: Daily stock prices for each company are retrieved from Yahoo Finance, adjusted for daily percentage changes, and normalized by the Dow Jones movement. The relationship between stock price fluctuations and tweet sentiment scores is explored to establish correlations. A predictive model is then developed and validated, forecasting stock movements from sentiment data. Predictions are stored in a file named ##-predict.txt (where ## is the company name), listing the date, predicted stock movement as a decimal, and a trading recommendation (sell as -1, hold as 0, buy as 1).



## Usage
- Run GenerateSentiment.ipynb for each desired day which will result in a "SentimentScoresAverage.xlsx" (we have already done this.)
- Run StockAnalysis.R which will result in "Tesla-predict.txt". This will consist of recommended trading strategies from both the Linear Regression model and the Random Forest model in the following form under the rec column for each model:

    | Sell | Hold | Buy  |
    |------|------|------|
    |  -1  |   0  |   1  |

## Results

Using "Tesla" as the company of interest: 

| Model    | Linear Regression | Random Forest
|----------|-------------------|---------------
| Accuracy |      0.7096774    | 0.8571429  

