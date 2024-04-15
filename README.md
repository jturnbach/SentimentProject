# Sentiment-Driven Forecasts: Analyzing Social Media's Influence on Stock Market Trends

## Introduction
This project examines the influence of social media sentiment on stock prices, using Twitter data to predict market trends through sentiment analysis and machine learning models.

## Data

The data was collected using the Tweepy and getOldTweets python libraries (not accessible with recent API changes.) Using a keyword search "Tesla" 10,000 tweets were collected over 50 days from March 6, 2020 to April 24, 2020.

## Usage
- Run GenerateSentiment.ipynb for each desired day which will result in a "SentimentScoresAverage.xlsx" (we have already done this.)
- Run StockAnalysis.R which will result in "Tesla-predict.txt"
