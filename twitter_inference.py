##### Preprocess functions #####
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO

def tweets_preprocess_backfill(df):

    # change datatype of date
    df.date = pd.to_datetime(df.date, utc=True)

    # drop nas
    df_clean = df.dropna()
    df_clean = df_clean.convert_dtypes()
    df_clean.user_followers = df_clean.user_followers.astype("float")
    df_clean.sort_values("date")

    Tags = ["Giveaway","giveaway","Cashback","cashback","Airdrop","nft"]
    # clear tweets with hastags that suggest bots and spam
    for tag in Tags:
        df_clean = df_clean[(df_clean.text.str.contains(tag)==False)]
    
    # data preprocessing. Remove hashtags, usernames, mentions, emojis, links, Non-ASCII characters, email adresses and special characters from posts
    # apply all the text cleaning functions
    df_clean['hashtag'] = df_clean.text.apply(func = hashtag_removal)
    df_clean['new_text'] = df_clean.text.apply(func = emoji_convert)
    df_clean['new_text'] = df_clean.new_text.apply(func = users_removal)
    df_clean['new_text'] = df_clean.new_text.apply(func = html_clean)
    df_clean['new_text'] = df_clean.new_text.apply(func = links_removal)
    df_clean['new_text'] = df_clean.new_text.apply(func = non_ascii_removal)
    df_clean['new_text'] = df_clean.new_text.apply(func = lower)
    df_clean['new_text'] = df_clean.new_text.apply(func = email_address_removal)
    df_clean['new_text'] = df_clean.new_text.apply(func = punct_removal)
    df_clean['new_text'] = df_clean.new_text.apply(func = special_removal)

    # only select tweets by users with more than 1000 followers
    df_influencers = df_clean[(df_clean.user_followers >= 1000)]
    # date to only be date, not time
    df_influencers.date = df_influencers.date.dt.floor('D')

    return df_influencers

def tweets_preprocess_daily(df):

    # change datatype of date
    df.date = pd.to_datetime(df.date, errors="coerce")

    # drop nas
    df_clean = df.dropna()
    df_clean = df_clean.convert_dtypes()
    df_clean.user_followers = df_clean.user_followers.astype("float")
    df_clean.sort_values("date")

    Tags = ["Giveaway","giveaway","Cashback","cashback","Airdrop","nft"]
    # clear tweets with hastags that suggest bots and spam
    for tag in Tags:
        df_clean = df_clean[(df_clean.text.str.contains(tag)==False)]
    
    # data preprocessing. Remove hashtags, usernames, mentions, emojis, links, Non-ASCII characters, email adresses and special characters from posts
    # apply all the text cleaning functions
    df_clean['hashtag'] = df_clean.text.apply(func = hashtag_removal)
    df_clean['new_text'] = df_clean.text.apply(func = emoji_convert)
    df_clean['new_text'] = df_clean.new_text.apply(func = users_removal)
    df_clean['new_text'] = df_clean.new_text.apply(func = html_clean)
    df_clean['new_text'] = df_clean.new_text.apply(func = links_removal)
    df_clean['new_text'] = df_clean.new_text.apply(func = non_ascii_removal)
    df_clean['new_text'] = df_clean.new_text.apply(func = lower)
    df_clean['new_text'] = df_clean.new_text.apply(func = email_address_removal)
    df_clean['new_text'] = df_clean.new_text.apply(func = punct_removal)
    df_clean['new_text'] = df_clean.new_text.apply(func = special_removal)

    # only select tweets by users with more than 1000 followers
    df_influencers = df_clean[(df_clean.user_followers >= 1000)]
    # date to only be date, not time
    df_influencers.date = df_influencers.date.dt.floor('D')

    return df_influencers

###### functions for text cleaning #####

# remove hashtags
def hashtag_removal(text):
    hash = re.findall(r"#(\w+)", text)
    return hash

# translate emoji
def emoji_convert(text):
    for emot in UNICODE_EMOJI:
        if text == None:
            text = text
        else:
            text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",", "").replace(":", "").split()))
        return text

# remove retweet username and tweeted at @username
def users_removal(tweet):
    '''Takes a string and removes retweet and @user information'''
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) 
    # remove tweeted at
    return tweet

# remove links
def links_removal(tweet):
  '''Takes a string and removes web links from it'''
  tweet = re.sub(r'http\S+', '', tweet) # remove http links
  tweet = re.sub(r'bit.ly/\S+', '', tweet) # remove bitly links
  tweet = tweet.strip('[link]') # remove [links]
  return tweet

# remove html links
def html_clean(text):
  html = re.compile('<.*?>')#regex
  return html.sub(r'',text)

# remove non ascii character
def non_ascii_removal(s):
  return "".join(i for i in s if ord(i)<128)

# convert to lower case
def lower(text):
  return text.lower()

# remove email address
def email_address_removal(text):
  email = re.compile(r'[\w\.-]+@[\w\.-]+')
  return email.sub(r'',text)

# remove punctuation
def punct_removal(text):
  token=RegexpTokenizer(r'\w+')#regex
  text = token.tokenize(text)
  text= " ".join(text)
  return text 

def special_removal(tweet):
  tweet = re.sub('([_]+)', "", tweet)
  return tweet



###### Twitter API Tweepy inference ######
# Daily tweet scraper

# Importing the libraries
import os
import configparser
import tweepy
import csv
from datetime import datetime, timedelta
import hopsworks
from tqdm import tqdm
# connect to Hopsworks
project = hopsworks.login(api_key_value='U6PiDFwDVDQHP26X.XhXDZQ9QKiNwafhLh11PUntcyYW5Zp8aoXhoj1IJTGHDBu8owQJUKbFClHaehyMU')
# connect to dataset API
dataset_api = project.get_dataset_api()

def scrape_tweets_daily():

    # folder to load files into locally
    if not os.path.exists("twitter_bitcoin_sentiment_assets"):
        os.mkdir("twitter_bitcoin_sentiment_assets")

    # load config file from hopsworks
    downloaded_file_path = dataset_api.download(
        "KTH_lab1_Training_Datasets/twitter_bitcoin_sentiment/config.ini", local_path="./twitter_bitcoin_sentiment_assets/", overwrite=True)
    
    # Read the config file
    config = configparser.ConfigParser()
    config.read('./twitter_bitcoin_sentiment_assets/config.ini')

    # Read the values
    api_key = config['twitter']['api_key']
    api_key_secret = config['twitter']['api_key_secret']
    access_token = config['twitter']['access_token']
    access_token_secret = config['twitter']['access_token_secret']

    # Authenticate
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    # parameter wait_on_rate_limit=True waits when request limit is reached 
    api = tweepy.API(auth, wait_on_rate_limit=True)

    columns = ['id', 'user_followers', 'date', 'text', 'hashtags']

    import pytz
    utc=pytz.UTC
    now = datetime.now(tz=utc)
    today = datetime.today()

    from dateutil.relativedelta import relativedelta
    # set the number of hours for historical tweet collection
    delta = now - relativedelta(hours=24)
    yesterday = (today - timedelta(days=1)).date()

    data = []
    df = pd.DataFrame(data, columns=columns)
    request = 0

    downloaded_file_path = dataset_api.download(
        "KTH_lab1_Training_Datasets/twitter_bitcoin_sentiment/all_accounts_more_than_10000.csv", local_path="./twitter_bitcoin_sentiment_assets/", overwrite=True)
    
    with open("./twitter_bitcoin_sentiment_assets/all_accounts_more_than_10000.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        print("Running through influential users...")
        for row in tqdm(reader,total=9464):
            value = int(row['followers'])
            # get tweets from users which have more than 500 000 followers
            if (value >= 500000): 
                # get two tweets from the user
                limit = 2
                repeat = 1
                try:
                    while(repeat == 1):
                        data2 = []
                        for tweet in tweepy.Cursor(api.user_timeline, user_id=row['id'], tweet_mode="extended").items(limit):
                            request += 1
                            if (tweet.created_at.date() >= yesterday):
                                text = tweet.full_text.replace("\n", " " )
                                data2.append([row['id'], row['followers'], tweet.created_at, text, tweet.entities.get('hashtags')])
                            latest = tweet.created_at.date()    
                            df2 = pd.DataFrame(data2, columns=columns)

                        # if the last tweet was released in the predifined period, then get more tweets from the same user
                        if(latest >= yesterday and limit < 32):
                            limit+=10 
                        else:
                            frames = [df, df2]
                            df = pd.concat(frames)
                            break
                except tweepy.errors.TweepyException as e:
                    pass

    df = df[(df['hashtags'].str.contains('Bitcoin')==True)|(df['text'].str.contains('Bitcoin')==True)|(df['hashtags'].str.contains('bitcoin')==True)|(df['text'].str.contains('bitcoin')==True)]
    return df