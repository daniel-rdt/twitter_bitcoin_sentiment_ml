import os
from unicodedata import decimal
import modal
from twitter_inference import tweets_preprocess_daily, scrape_tweets_daily, tweets_preprocess_backfill

BACKFILL = True
LOCAL = True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image","tweepy","datetime","configparser","transformers"]) 

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
    from textblob import TextBlob


    # connect to Hopsworks
    project = hopsworks.login(api_key_value='U6PiDFwDVDQHP26X.XhXDZQ9QKiNwafhLh11PUntcyYW5Zp8aoXhoj1IJTGHDBu8owQJUKbFClHaehyMU')
    # connect to feature store
    fs = project.get_feature_store()
    # connect to dataset API
    dataset_api = project.get_dataset_api()

    ##### functions for sentiment analysis #####

    # create a function to get the subjectivity
    def getSubjectivity(twt):
        return TextBlob(twt).sentiment.subjectivity

    # create a function to get the polarity
    def getPolarity(twt):
        return TextBlob(twt).sentiment.polarity

    ##### Backfill #####

    # either use prepped tweets dataset reaching back to 2016 to add to feature group
    if BACKFILL == True:
        print("Backfill started. Downloading twitter backfill file...")
        # load file from hopsworks api
        if not os.path.exists("twitter_bitcoin_sentiment_assets"):
            os.mkdir("twitter_bitcoin_sentiment_assets")
        downloaded_file_path = dataset_api.download(
            "KTH_lab1_Training_Datasets/twitter_bitcoin_sentiment/tweets_influential_users.csv", local_path="./twitter_bitcoin_sentiment_assets/", overwrite=True)
        df = pd.read_csv("./twitter_bitcoin_sentiment_assets/tweets_influential_users.csv", sep=";", decimal=".",lineterminator='\n',usecols=['time', 'id', 'tweet','followers'], index_col='id')
        df = df.rename(columns={'time':'date','tweet':'text','followers':'user_followers'})
        print(df.info())
        print("Finished twitter backfill file download. Starting preprocess...")
        twitter_df = tweets_preprocess_backfill(df)
        print("Extracting tweet sentiment...")
        # extract sentiment from tweet text
        # set up pipe from huggingface transformer pipeline
        from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
        from transformers import pipeline
        from scipy.special import softmax
        # to track progress of sentiment analysis
        from tqdm import tqdm, tqdm_pandas
        tqdm.pandas()

        print("Initializing transformers pipeline...")
        model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        def sentiment_analysis(text):
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            # determine ranking
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            ranked_label = config.id2label[ranking[0]]
            # # printing of result
            # for i in range(scores.shape[0]):
            #     l = config.id2label[ranking[i]]
            #     s = scores[ranking[i]]
            #     print(f"{i+1}) {l} {np.round(float(s), 4)}")

            # scores is [negative, neutral, positive]
            return scores

        print("Running pipeline (this can take up to 2h or more)...")
        twitter_df[['sentiment_score_negative','sentiment_score_neutral','sentiment_score_positive']] = pd.DataFrame(twitter_df.new_text.progress_apply(sentiment_analysis).tolist(), index= twitter_df.index)
        print("Pipeline finished!")

        twitter_df.to_csv("./twitter_bitcoin_sentiment_assets/checkpoint_with_sentiment.csv",sep=";",decimal=".")
        # only select relevant features
        twitter_df = twitter_df[["user_followers","date","new_text","sentiment_score_negative","sentiment_score_neutral","sentiment_score_positive"]]
        # group by days
        d = {'user_followers':'aggregate_followers','sentiment_score_negative':'sentiment_score_negative_mean','sentiment_score_neutral':'sentiment_score_neutral_mean','sentiment_score_positive':'sentiment_score_positive_mean'}
        twitter_df = twitter_df.groupby(pd.Grouper(key='date',freq='D')).agg({'user_followers':'sum','sentiment_score_negative':'mean','sentiment_score_neutral':'mean','sentiment_score_positive':'mean'}).rename(columns=d).dropna()

        try:
            twitter_df.index = twitter_df.index.tz_convert(None)
        except:
            pass
        print("Preprocess finished. With following snippet of twitter dataframe:")
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
        # categorize fluctuation as neutral, bearish or bullish when fluctuation is greater or less than 2% bullish or bearish, in between neutral
        bitcoin_df['bitcoin_fluctuation'] = 'Neutral'
        bitcoin_df.loc[((bitcoin_df.close - bitcoin_df.open)/bitcoin_df.open) > 0.02, 'bitcoin_fluctuation'] = 'Bullish'
        bitcoin_df.loc[((bitcoin_df.close - bitcoin_df.open)/bitcoin_df.open) < -0.02, 'bitcoin_fluctuation'] = 'Bearish'
        counts = np.unique(bitcoin_df.bitcoin_fluctuation, return_counts=True)[1]
        print(f"Bitcoin fluctuations are:\nBearish: {counts[0]}\nNeutral: {counts[2]}\nBullish: {counts[1]}")
        bitcoin_df.set_index("date", inplace=True)
        bitcoin_df_input = bitcoin_df[["bitcoin_fluctuation"]]
        twitter_df = twitter_df.merge(bitcoin_df_input, on='date')
        print("Merged with following snippet of final dataframe:")
        print(twitter_df.head())

    # or update with new twitter and corresponding bitcoin fluctuation data from yesterday
    else:
        print("Started daily update. Scraping daily tweets from Bitstamp API...")
        # scrape yesterdays tweets from users which have more than 500 000 followers
        tweets_df = scrape_tweets_daily()
        # tweets_df = pd.read_csv("project/twitter_bitcoin_sentiment_ml/Tomas_files/220103_tweets_bitcoin.csv", decimal=".",sep=";", index_col=0)
        print("Finished scraping daily tweets. Starting preprocess...")
        # iloc[0] to only select yesterday's tweets and not today's
        twitter_df = tweets_preprocess_daily(tweets_df).iloc[[0]]
        print("Extracting tweet sentiment...")
        # extract sentiment from tweet text
        # set up pipe from huggingface transformer pipeline
        from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
        from transformers import pipeline
        from scipy.special import softmax
        # to track progress of sentiment analysis
        from tqdm import tqdm, tqdm_pandas
        tqdm.pandas()

        print("Initializing transformers pipeline...")
        model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        def sentiment_analysis(text):
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            # determine ranking
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            ranked_label = config.id2label[ranking[0]]
            # # printing of result
            # for i in range(scores.shape[0]):
            #     l = config.id2label[ranking[i]]
            #     s = scores[ranking[i]]
            #     print(f"{i+1}) {l} {np.round(float(s), 4)}")

            # scores is [negative, neutral, positive]
            return scores

        print("Running pipeline (this can take up to 2h or more)...")
        twitter_df[['sentiment_score_negative','sentiment_score_neutral','sentiment_score_positive']] = pd.DataFrame(twitter_df.new_text.progress_apply(sentiment_analysis).tolist(), index= twitter_df.index)
        print("Pipeline finished!")

        twitter_df.to_csv("./twitter_bitcoin_sentiment_assets/checkpoint_with_sentiment.csv",sep=";",decimal=".")
        # only select relevant features
        twitter_df = twitter_df[["user_followers","date","new_text","sentiment_score_negative","sentiment_score_neutral","sentiment_score_positive"]]
        # group by days
        d = {'user_followers':'aggregate_followers','sentiment_score_negative':'sentiment_score_negative_mean','sentiment_score_neutral':'sentiment_score_neutral_mean','sentiment_score_positive':'sentiment_score_positive_mean'}
        twitter_df = twitter_df.groupby(pd.Grouper(key='date',freq='D')).agg({'user_followers':'sum','sentiment_score_negative':'mean','sentiment_score_neutral':'mean','sentiment_score_positive':'mean'}).rename(columns=d).dropna()

        try:
            twitter_df.index = twitter_df.index.tz_convert(None)
        except:
            pass
        print("Preprocess finished with following snipped of twitter dataframe:")
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
        bitcoin_df['bitcoin_fluctuation'] = 'Neutral'
        bitcoin_df.loc[((bitcoin_df.close - bitcoin_df.open)/bitcoin_df.open) > 0.02, 'bitcoin_fluctuation'] = 'Bullish'
        bitcoin_df.loc[((bitcoin_df.close - bitcoin_df.open)/bitcoin_df.open) < -0.02, 'bitcoin_fluctuation'] = 'Bearish'
        counts = np.unique(bitcoin_df.bitcoin_fluctuation, return_counts=True)[1]
        print(f"Bitcoin fluctuations are:\nBearish: {counts[0]}\nNeutral: {counts[2]}\nBullish: {counts[1]}")
        bitcoin_df.set_index("date", inplace=True)
        bitcoin_df_input = bitcoin_df[["bitcoin_fluctuation"]]
        twitter_bitcoin_df = twitter_df.merge(bitcoin_df_input, on='date')
        print("Merged with following snippet of final dataframe:")
        print(twitter_bitcoin_df.head())
    
    # add to feature group
    print("Pushing to feature group...")
    twitter_fg = fs.get_or_create_feature_group(
        name="twitter_bitcoin_sentiment",
        version=1,
        primary_key=['aggregate_followers','sentiment_score_negative_mean','sentiment_score_neutral_mean','sentiment_score_positive_mean'], 
        description="Twitter bitcoin sentiment dataset")
    twitter_fg.insert(twitter_df, write_options={"wait_for_job" : False})
    print("Feature pipeline finished!")

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("bitcoin_sentiment_feature_pipeline_daily")