import gradio as gr
import imageio.v3 as iio
from PIL import Image
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/latest_bitcoin_fluctuation_prediction_2.png", overwrite=True)
dataset_api.download("Resources/images/latest_bitcoin_fluctuation_actual_2.png", overwrite=True)
dataset_api.download("Resources/images/df_recent_2.png", overwrite=True)
dataset_api.download("Resources/images/confusion_matrix_2.png", overwrite=True)

def update():
    dataset_api.download("Resources/images/latest_bitcoin_fluctuation_prediction_2.png", overwrite=True)
    dataset_api.download("Resources/images/latest_bitcoin_fluctuation_actual_2.png", overwrite=True)
    dataset_api.download("Resources/images/df_recent_2.png", overwrite=True)
    dataset_api.download("Resources/images/confusion_matrix_2.png", overwrite=True)

def update_fluctuation_prediction_img():
    im_pred = iio.imread('latest_bitcoin_fluctuation_prediction_2.png')
    return im_pred

def update_actual_fluctuation_img():
    im_act = iio.imread('latest_bitcoin_fluctuation_actual_2.png')
    return im_act

def update_df_recent_img():
    im_hist = iio.imread('df_recent_2.png')
    return im_hist

def update_confusion_matrix_img():
    im_matr = iio.imread('confusion_matrix_2.png')
    return im_matr

with gr.Blocks() as demo:
    with gr.Row():
      gr.Markdown(
        """
        # Bitcoin Twitter Sentiment Predictor Monitor v2
        Model version build upon hyperparameter tuning for max. f-1 score.
        """
        )
    with gr.Row():
        load=gr.Button("Load Images")
        load.click(fn=update)
    with gr.Row():
        refresh=gr.Button("Refresh (wait 10 seconds after loading images before refreshing")
        
    with gr.Row():
      with gr.Column():
        gr.Label("Today's Predicted Image")
        input_img_pred = gr.Image("latest_bitcoin_fluctuation_prediction_2.png", elem_id="predicted-img")
        refresh.click(update_fluctuation_prediction_img,outputs=input_img_pred)

      with gr.Column():          
        gr.Label("Today's Actual Image")
        input_img_act = gr.Image("latest_bitcoin_fluctuation_actual_2.png", elem_id="actual-img")
        refresh.click(update_actual_fluctuation_img,outputs=input_img_act)
        
    with gr.Row():
      with gr.Column():
        gr.Label("Recent Prediction History")
        input_img_hist = gr.Image("df_recent_2.png", elem_id="recent-predictions")
        refresh.click(update_df_recent_img,outputs=input_img_hist)

      with gr.Column():          
        gr.Label("Confusion Maxtrix with Historical Prediction Performance")
        input_img_matr = gr.Image("confusion_matrix_2.png", elem_id="confusion-matrix")  
        refresh.click(update_confusion_matrix_img,outputs=input_img_matr)
      

demo.launch()
