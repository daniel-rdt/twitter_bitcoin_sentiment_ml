import os
from pickle import TRUE
import modal
from twitter_inference import tweets_preprocess_backfill

BACKFILL = True
LOCAL = True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image"]) 

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

    # either use whole prepped dataset as in titanic-feature-pipeline to add to feature group
    if BACKFILL == True:
        df = pd.read_csv("./project/twitter_bitcoin_sentiment_ml/Kaggle_Bitcoin_tweets.csv",
                        usecols=[
                             "user_name",
                             "user_followers",
                             "user_verified",
                             "date",
                             "text",
                             "hashtags"],
                        dtype={
                             "user_name": str,
                             "user_location": str,
                             "user_description": str,
                             "text": str},
                         parse_dates=['date'],
                        )
        twitter_df = tweets_preprocess_backfill(df)
        # import bitcoin price data to match with tweets from previous day
        bitcoin_df = pd.read_csv("project/twitter_bitcoin_sentiment_ml/bitcoin_bitstamp.csv", decimal=".", usecols=[
                             "date",
                             "symbol",
                             "open",
                             "high",
                             "low",
                             "close"],
                         parse_dates=['date'])
        # date to only be date, not time
        bitcoin_df.date = bitcoin_df.date.dt.floor('D')
        bitcoin_df["Bitcoin_Fluctuation_nextday"] = np.where((bitcoin_df.open - bitcoin_df.close) < 0, "bearish", "bullish")
        # subtract one day so when matching with tweets bitcoin fluctuation of next day will be the label for the tweets of a given day
        bitcoin_df.date = bitcoin_df.date - timedelta(days=1)
        bitcoin_df.set_index("date", inplace=True)
        bitcoin_df_input = bitcoin_df[["Bitcoin_Fluctuation_nextday"]]
        twitter_df.join(bitcoin_df_input)
        print(twitter_df)

    # or use newly created passenger to add to feature group
    # else:
    #     twitter_df = tweets_preprocess_daily()

    # add to feature group
    twitter_fg = fs.get_or_create_feature_group(
        name="twitter_bitcoin_sentiment_model",
        version=1,
        primary_key=["aggregate_followers","subjectivity_mean","polarity_mean"], 
        description="Twitter bitcoin sentiment dataset")
    twitter_fg.insert(twitter_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("feature_pipeline_daily")