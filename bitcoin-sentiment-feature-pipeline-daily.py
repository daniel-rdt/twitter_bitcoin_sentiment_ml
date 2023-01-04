import os
import modal
from twitter_inference import tweets_preprocess_daily, scrape_tweets_daily, tweets_preprocess_backfill

BACKFILL = True
LOCAL = True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image","tweepy","datetime","configparser"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("hopsworks-api-key"))
   def f():
       g()


def g():
    import hopsworks
    import pandas as pd
    import numpy as np
    import tweepy
    import datetime
    from datetime import date, timedelta
    import time
    import configparser

    project = hopsworks.login()
    fs = project.get_feature_store()

    # either use prepped tweets dataset reaching back to 2016 to add to feature group
    if BACKFILL == True:
        print("Backfill started. Downloading twitter backfill file...")
        df = pd.read_csv('project/scraped_tweets/tweets_influential_users.csv', sep=';', decimal='.', lineterminator='\n')
        df = df.rename(columns={'time':'date','tweet':'text','followers':'user_followers'})
        print("Finished twitter backfill file download. Starting preprocess...")
        twitter_df = tweets_preprocess_backfill(df)
        print("Preprocess finished. With following snippet of twitter dataframe:")
        try:
            twitter_df.index = twitter_df.index.tz_convert(None)
        except:
            pass
        print(twitter_df.head())
        print("Downloading Bitcoin data...")
        # import bitcoin price data to match with tweets from that day
        bitcoin_df = pd.read_csv('https://www.cryptodatadownload.com/cdd/Bitstamp_BTCEUR_d.csv', skiprows=1, decimal=".", usecols=[
                             "date",
                             "symbol",
                             "open",
                             "high",
                             "low",
                             "close"],
                         parse_dates=['date'])
        print("Bitcoin data downloaded. Processing and merging with twitter data...")
        # date to only be date, not time
        bitcoin_df.date = bitcoin_df.date.dt.floor('D')
        bitcoin_df["Bitcoin_Fluctuation"] = np.where((bitcoin_df.open - bitcoin_df.close) < 0, "Bearish", "Bullish")
        bitcoin_df.set_index("date", inplace=True)
        bitcoin_df_input = bitcoin_df[["Bitcoin_Fluctuation"]]
        twitter_df = twitter_df.merge(bitcoin_df_input, on='date')
        print("Merged with following snippet of final dataframe:")
        print(twitter_df.head())

    # or update with new twitter and corresponding bitcoin fluctuation data from yesterday
    else:
        print("Started daily update. Scraping daily tweets from Bitstamp API...")
        # scrape yesterdays tweets from users which have more than 500 000 followers
        # tweets_df = scrape_tweets_daily()
        tweets_df = pd.read_csv("project/twitter_bitcoin_sentiment_ml/Tomas_files/220103_tweets_bitcoin.csv", decimal=".",sep=";", index_col=0)
        print("Finished scraping daily tweets. Starting preprocess...")
        # iloc[0] to only select yesterday's tweets and not today's
        twitter_df = tweets_preprocess_daily(tweets_df).iloc[[0]]
        try:
            twitter_df.index = twitter_df.index.tz_convert(None)
        except:
            pass
        print(" Preprocess finished with following snipped of twitter dataframe:")
        print(twitter_df.head())
        print("Downloading Bitcoin data...")
        # import bitcoin price data to match with tweets from that day
        bitcoin_df = pd.read_csv('https://www.cryptodatadownload.com/cdd/Bitstamp_BTCEUR_d.csv', skiprows=1, decimal=".", usecols=[
                             "date",
                             "symbol",
                             "open",
                             "high",
                             "low",
                             "close"],
                         parse_dates=['date'])
        print("Bitcoin data downloaded. Processing and merging with twitter data...")
        # date to only be date, not time
        bitcoin_df.date = bitcoin_df.date.dt.floor('D')
        bitcoin_df["Bitcoin_Fluctuation"] = np.where((bitcoin_df.open - bitcoin_df.close) < 0, "Bearish", "Bullish")
        bitcoin_df.set_index("date", inplace=True)
        bitcoin_df_input = bitcoin_df[["Bitcoin_Fluctuation"]]
        twitter_bitcoin_df = twitter_df.merge(bitcoin_df_input, on='date')
        print("Merged with following snippet of final dataframe:")
        print(twitter_bitcoin_df.head())
    
    # add to feature group
    print("Pushing to feature group...")
    twitter_fg = fs.get_or_create_feature_group(
        name="twitter_bitcoin_sentiment",
        version=1,
        primary_key=["aggregate_followers","subjectivity_mean","polarity_mean"], 
        description="Twitter bitcoin sentiment dataset")
    twitter_fg.insert(twitter_df, write_options={"wait_for_job" : False})
    print("Feature pipeline finished!")

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("bitcoin_sentiment_feature_pipeline_daily")