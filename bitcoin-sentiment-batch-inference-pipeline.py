import os
import modal
    
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","scikit-learn","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("hopsworks-api-key"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    # get model from model registry from hopswork
    mr = project.get_model_registry()
    model = mr.get_best_model("twitter_bitcoin_sentiment", 'accuracy', 'max')
    model_dir = model.download()
    model = joblib.load(model_dir + "/twitter_bitcoin_sentiment_model.pkl")
    
    # get batch data from hopsworks feature view
    feature_view = fs.get_feature_view(name="twitter_bitcoin_sentiment", version=1)
    batch_data = feature_view.get_batch_data()
    
    # make prediction on whole batch data set (returns encoded label (le) that needs to be inverse transformed)
    y_pred_le = model.predict(batch_data)
    # label encoder and transform
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(['Bearish', 'Bullish','Neutral'])
    # print(list(le.classes_))
    y_pred = le.inverse_transform(y_pred_le)
    # print(y_pred)
    
    # determine outcome of the latest prediction and download the appropriate image from GitHub
    fluctuation = y_pred[-1]
    prediction_url = f"https://raw.githubusercontent.com/daniel-rdt/twitter_bitcoin_sentiment_ml/main/assets/{fluctuation}.jpg"
    
    # print the prediction in console
    print(f"Predicted bitcoin fluctuation: {y_pred}")

    # save image in dataset api
    img = Image.open(requests.get(prediction_url, stream=True).raw)            
    img.save("./latest_bitcoin_fluctuation_prediction.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_bitcoin_fluctuation_prediction.png", "Resources/images", overwrite=True)
    
    # get feature group and get latest passenger
    twitter_fg = fs.get_feature_group(name="twitter_bitcoin_sentiment", version=1)
    df = twitter_fg.read()
    # print(df.iloc[-1])

    # determine actual label of passenger and download the appropriate image from GitHub
    label = df.iloc[-1]["bitcoin_fluctuation"]
    label_url = f"https://raw.githubusercontent.com/daniel-rdt/twitter_bitcoin_sentiment_ml/main/assets/{label}.jpg"
    
    print(f"Actual bitcoin fluctuation: {label}!")

    # save the image in dataset api
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./latest_bitcoin_fluctuation_actual.png")
    dataset_api.upload("./latest_bitcoin_fluctuation_actual.png", "Resources/images", overwrite=True)
    
    # get prediction feature group from hopsworks or create new one
    monitor_fg = fs.get_or_create_feature_group(name="twitter_bitcoin_sentiment_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Twitter bitcoin sentiment Prediction/Outcome Monitoring"
                                                )
    
    # create datetetime information for prediction
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [fluctuation],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    # insert newest prediction to predictions feature group
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    # get last 5 predictions and store image on dataset api
    model_dir="twitter_bitcoin_sentiment_predictions"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)
    df_recent = history_df.tail(5)
    dfi.export(df_recent, 'twitter_bitcoin_sentiment_predictions/df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("twitter_bitcoin_sentiment_predictions/df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when there are examples of both bearish and bullish behaviour predicted
    print("Number of different bitcoin fluctuation predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions, labels=['Bearish', 'Bullish'])

    
        df_cm = pd.DataFrame(results, ['True Bearish', 'True Bullish'],
                            ['Pred Bearish', 'Pred Bullish'])

        cm = sns.heatmap(df_cm, annot=True, fmt='g')
        fig = cm.get_figure()
        fig.savefig("twitter_bitcoin_sentiment_predictions/confusion_matrix.png")
        # save confusion matrix image to dataset api
        dataset_api.upload("twitter_bitcoin_sentiment_predictions/confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different bitcoin fluctuation predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different bitcoin fluctuation predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("bitcoin_sentiment_batch_inference_pipeline_daily")
