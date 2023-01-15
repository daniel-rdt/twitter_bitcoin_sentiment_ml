# Bitcoin Performance prediction using twitter sentiment analysis

Over the last decade cryptocurrencies such as Bitcoin have become of great interest on the financial markets. 
A big topic herein has been the huge fluctuations of Bitcoin prices from day to day as it is subject to speculation and the public's opinion on Bitcoin.
Therefore, it is increasingly hard to predict how the Bitcoin will perform on a daily basis.

For this reason, this project aims at using twitter sentiments regarding Bitcoin to create a machine learning model that predicts whether the Bitcoin will behave bullish, bearish or neutral on a daily basis.
The idea is to use tweets with mention of Bitcoin or respective synonyms and applying an already existing natural language processing model to pre-process the input data. From that a xgBoost classification algorithm is applied to predict the bitcoin fluctuation label of a given day.

Scripts:
1.    bitcoin-sentiment-feature-pipeline-daily.py

2.    twitter_inference.py (helper functions for inference with twitter data)

3.1.  bitcoin-sentiment-training-pipeline.py

3.2.  bitcoin-sentiment-training-pipeline-2.py

4.1.  bitcoin-sentiment-batch-inference-pipeline.py

4.2.  bitcoin-sentiment-batch-inference-pipeline-2.py

5.1.  huggingface-spaces-bitcoin-sentiment-monitor/app.py

5.2.  huggingface-spaces-bitcoin-sentiment-monitor-2/app.py

The feature and batch inference pipelines can either be run locally or using serverless compute platforms on a determined run-schedule such as Modal. The training pipeline should be run on demand. For scalability, the pipelines are separated and a feature store is utilized in the form of Hopsworks Feature Store. The predictions can be monitored using a public Huggingface App. Furthermore, two approaches were taken for the model tuning. Maximising accuracy and maximising F1-score, yielding very different results. Thus, both approaches were given a training and scheduled batch pipeline as well as a separate monitor on Huggingface.

## Influential accounts dataset

For daily data feeding the Twitter API was used. It can be found here: https://www.tweepy.org. We got access to premium features such as historical data and higher scrapping limits.
Tweepy has a lot of limits concerning tweet scraping. In order get more tweets, we chose to scrap tweets from individual account ID's. The first step was to get such accounts, so we did some research in Google. The collection of account names were taken from websites such as: https://coinme.com/10-crypto-twitter-accounts-everyone-should-follow/ and https://learn.bybit.com/crypto/best-crypto-twitter-accounts/. Then we ran a script which converted account names to ID's. In order to make a bigger accounts dataset, we ran a script which extracted what these accounts are following. When we got new dataset, we ran again what those accounts are following. We ran a scipt which scrapes newest tweets with the keyword `bitcoin` from the whole Twitter. After running for a few days, we got a new dataset of tweets. Then we took all the accounts ID's from this dataset. Finally we got a dataset of 9464 users. This dataset is in all_accounts_more_than_10000.csv file.

## Tweet dataset
We wrote the script to collect Tweet dataset. We ran that script over all the accounts which are in all_accounts_more_than_10000.csv file and got 400 Tweets from each account. In total we have 3 785 600 Tweets which dates about 30 days back up to 31 of December.

## Feature Pipeline: bitcoin-sentiment-feature-pipeline-daily.py
This file has backfill mode and inserts these functions from the helper function py-file twitter_inference.py: tweets_preprocess_daily, scrape_tweets_daily, tweets_preprocess_backfill
It can be run either locally or with MODAL on a daily schedule.
The tweets are preprocessed and a huggingface transformers model based on the large pre-trained language model roBERTa is applied to every Tweet's text. The text gets three scores: sentiment_positive, sentiment_negative, sentiment_neutral that are between 0 and 1 and add up to 1. The sentiment model used is `Twitter-roBERTa-base for Sentiment Analysis - UPDATED (2022)`and can be found here: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

The Bitcoin price data is used from Bitstamp exchange: https://www.cryptodatadownload.com/data/bitstamp/?fbclid=IwAR1k1Lbo-Bz6ocD85Y7ex17qcWs6bIpqnSZQbI3kTYyOBTmQWh9Hkt3Plf0#google_vignette.
The Bitstamp dataset includes historical OHLC price data on a daily, hourly, and minute basis for the spot and physical market. The database is updated daily and is openly accessible.

In this work we assume that Bitcoin trend is bullish or bearish if the difference in opening and closing price is bigger or lower than 2 % accordingly. In other case it is neutral. This data is used as a labels.

The prepared dataset is then aggregated to daily rows and uploaded to `Hopsworks` to be downloaded later in the training pipeline.

## twitter_inference.py
Function tweets_preprocess_backfill is for backfill data preprocessing and tweets_preprocess_daily is for daily preprocessing. Feature engineering in those functions:
1. data type change
2. dropped NAs
3. teets sorted by time. 
4. tweets with hashtags containing keywords such as `Giveaway`, `giveaway`, `Cashback`, `cashback`, `Airdrop` and `nft` were removed.
5. enojis converted, user mentions removed, links removed, html cleaned, non ascii removed, text converted to lower case, email address removed, punctuation removed, special signs removed. 
6. tweeets written by accounts which have less than 1000 followers were removed.
7. tweet time were reduced to days.

Function scrape_tweets_daily scrapes tweets from 9464 accounts which are in the .csv file. Due to certain API limitations in the beginning we take 2 tweets and if the second tweet is not from yesterday, we take again from the beginning 12 tweets from the same account timeline. If the 12th tweet is not from yesterday, we take again from the beginning 22 tweets from the same account timeline. If the 22nd tweet is not from yesterday, we take again from the beginning 32 tweets from the same account timeline.

We collect tweet creation date, tweet text and hashtags. 
Also the script only leaves those Tweets which have `Bitcoin`or related keyword.

## Training Pipeline: bitcoin-sentiment-training-pipeline.py
A classification algorithm XGBoost was applied. One day of tweet input data was aggregated into one input into the model yielding one prediction.
To calculate the labels, Bitcoin market prices are used and opening and closing prices are compared to create either bullish, bearish or neutral label for Bitcoin behavior of the respective day.

A number of times the model was adjusted, executed and the best hyperparameters were found. Fine-tuned model parameters: learning_rate: 0.01, max_depth: 3, min_child_weight: 1, n_estimators: 200.

## Batch Inference Pipeline: bitcoin-sentiment-batch-inference-pipeline.py
- Can be run either locally or with MODAL on a daily schedule;
- gets model from Hopsworks registry and batch data from feature view;
- predicts if the Bitcoin trend is bullish, bearish or neutral;
- downloads appropriate images for prediction and creates up to date confusion matrix when there have been 3 predictions made to date;
- stores predictions to Hopsworks;

## UI Inference: huggingface-spaces-bitcoin-sentiment-monitor/app.py

Folder contains main app for user interface and requirements.

The UI Inference can be found here: https://huggingface.co/spaces/daniel-rdt/twitter-bitcoin-sentiment

The aim of the UI is to demonstrate past predictions in a table, confusion matrix, show today's actual and predicted bitcoin trend and also to show the bitcoin trend for the last 7 days. The refresh button is created to update results manualy.

## Results and final thoughts
Overall the accuracy of the model is 59.1%. To improve this measure a bigger dataset of Twitter users and tweeets should be used, especially over a longer period of time that experienced more bullish and bearish fluctuations. In addition, it is important to take in consideration Bitcoin graph technical analysis and stock market. For example, Bitcoin is tightly related to S&P500 index.
The project took way longer to complete than we expected. The hardest part and the most time consuming was Twitter API because it took some time to get approval to use it from Twitter and because of API limitation. The dataset collection of Tweets and accounts were also very time consuming and took about few weeks. After getting all data everything else (feature engineering, training...) took about the same amount of time to complete. 
