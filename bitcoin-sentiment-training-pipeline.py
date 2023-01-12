import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks", "seaborn", "joblib", "scikit-learn","xgboost"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("hopsworks-api-key"))
   def f():
       g()


def g():
    import hopsworks
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import seaborn as sns
    from matplotlib import pyplot
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib

    project = hopsworks.login(api_key_value='U6PiDFwDVDQHP26X.XhXDZQ9QKiNwafhLh11PUntcyYW5Zp8aoXhoj1IJTGHDBu8owQJUKbFClHaehyMU')
    # get feature store from hopsworks
    fs = project.get_feature_store()

    # select or create feature view from feature group of titanic dataset in feature store
    try: 
        feature_view = fs.get_feature_view(name="twitter_bitcoin_sentiment", version=1)
    except:
        twitter_fg = fs.get_feature_group(name="twitter_bitcoin_sentiment", version=1)
        query = twitter_fg.select_all()
        feature_view = fs.create_feature_view(name="twitter_bitcoin_sentiment",
                                          version=1,
                                          description="Read from Twitter bitcoin sentiment dataset",
                                          labels=["bitcoin_fluctuation"],
                                          query=query)    

    # Read training data, randomly split into train/test sets of features (X) and labels (y) with 80/20 split      
    X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)
    # transform labels into category and transform with label encoder from scikit learn
    y_train.bitcoin_fluctuation = y_train.bitcoin_fluctuation.astype('category')
    y_test.bitcoin_fluctuation = y_test.bitcoin_fluctuation.astype('category')
    # label encoder and transform
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(['Bearish', 'Bullish','Neutral'])
    # print(list(le.classes_))
    y_train_le = le.transform(y_train.values.ravel())

    # Train model with the sklearn API of xgBoost algorithm using our features (X_train) and our encoded labels (y_train_le)

    # # First: Hyperparameter tuning using grid search cross validation
    # from sklearn.model_selection import GridSearchCV
    # # set ranges for hyperparamers 'learning_rate' (eta), 'max_depth', 'min_child_weight', 'n_estimators'
    # grid_param = {
    #     'learning_rate': [0.01,0.02,0.03,0.04,0.05,0.1,0.2], 
    #     'max_depth': [3,4,5,6,7,8,9,10],
    #     'min_child_weight': [0.5,1,2],
    #     'n_estimators': [200,400,800,1000,1500]
    # }
    # # initialize grid search cv object
    # gd_sr = GridSearchCV(estimator=xgb.XGBClassifier(tree_method="hist", objective="multi:softmax"),
    #                  param_grid=grid_param,
    #                  scoring='accuracy',
    #                  cv=5,
    #                  n_jobs=-1)
    # # run grid seatch on training data
    # gd_sr.fit(X_train,y_train_le)
    # # check best parameters and resultung accuracy
    # best_parameters = gd_sr.best_params_
    # print(best_parameters)
    # best_result = gd_sr.best_score_
    # print(best_result)
    # # resulting best parameters are: {'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 200} with a cv accuracy score of 0.561

    # Now train model with found hyperparameters
    model = xgb.XGBClassifier(tree_method="hist", learning_rate=0.01, n_estimators=200, max_depth=3, min_child_weight=1, objective="multi:softmax")
    model.fit(X_train, y_train_le)

    # Evaluate model performance using the features from the test set (X_test)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Compare predictions (y_pred) with the labels in the test set (y_test)
    metrics = classification_report(y_test, le.inverse_transform(y_pred), output_dict=True)
    metrics_train = classification_report(y_train, le.inverse_transform(y_pred_train), output_dict=True)
    print(f"Accuracy Test: {metrics['accuracy']}")
    print(f"Accuracy Train: {metrics_train['accuracy']}")
    
    # Create the confusion matrix as a figure
    results = confusion_matrix(y_test, le.inverse_transform(y_pred),labels=['Bearish', 'Bullish', 'Neutral'])
    df_cm = pd.DataFrame(results, ['True Bearish', 'True Bullish', 'True Neutral'],
                            ['Pred Bearish', 'Pred Bullish', 'Pred Neutral'])
    cm = sns.heatmap(df_cm, annot=True, fmt='g')
    fig = cm.get_figure()

    # Upload model to the Hopsworks Model Registry
    # First get an object for the model registry.
    mr = project.get_model_registry()
    
    # The contents of the 'titanic_model' directory will be saved to the model registry. Create the dir, first.
    model_dir="twitter_bitcoin_sentiment_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)

    # Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
    joblib.dump(model, model_dir + "/twitter_bitcoin_sentiment_model.pkl")
    fig.savefig(model_dir + "/confusion_matrix.png")    


    # Specify the schema of the model's input/output using the features (X_train) and transformed labels (y_train_le)
    input_schema = Schema(X_train)
    output_schema = Schema(y_train_le)
    model_schema = ModelSchema(input_schema, output_schema)

    # Create an entry in the model registry that includes the model's name, desc, metrics
    twitter_bitcoin_sentiment_model = mr.python.create_model(
        name="twitter_bitcoin_sentiment", 
        metrics={"accuracy" : metrics['accuracy']},
        model_schema=model_schema,
        description="Twitter Bitcoin Sentiment Predictor"
    )
    
    # Upload the model to the model registry, including all files in 'model_dir'
    twitter_bitcoin_sentiment_model.save(model_dir)
    
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
