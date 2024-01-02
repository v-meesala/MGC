import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.src.layers import Embedding, Conv1D, GlobalMaxPooling1D, Add, concatenate, TextVectorization
import tensorflow as tf
import joblib
import os
from tensorflow.python.keras.models import load_model
from fastapi import FastAPI
import utils
import ScrapeLyrics as sl
import extractAudioFeatures as eaf
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
# importing packages
from pytube import YouTube
import os
from bs4 import BeautifulSoup
import requests
from googleapiclient.discovery import build

app = FastAPI()


def modelPrediction(audio_features_array, lyrics_array):

    # Final model
    model = keras.models.load_model('my_model.keras')

    # Predict the genre
    genre_prediction = model.predict([audio_features_array, lyrics_array])

    # If you want the index of the highest probability which represents the predicted class
    predicted_genre_index = np.argmax(genre_prediction, axis=1)

    #print max probability
    print(f"The max probability is: {np.max(genre_prediction)}")

    # Load the saved LabelEncoder
    label_encoder = joblib.load('label_encoder.joblib')

    # Assuming `predicted_genre_index` is the output index from your model's prediction
    predicted_genre_name = label_encoder.inverse_transform(predicted_genre_index)

    print(f"The predicted genre is: {predicted_genre_name[0]}")

    return predicted_genre_name[0]


def findGenre(title, artist):
    #get current project path
    currentPath = os.getcwd()
    #in current path go to Datasets/2018655052_small
    filepath = os.path.join(currentPath, 'Tests')

    fname = str(artist) + '-' + str(title) + '.mp3'

    #get audio file from the above filepath
    filename  = os.path.join(filepath, fname)


    #Extracting audio features
    audio_features_array = eaf.getAudioFts(filename)
    #reshape as 1row numpy array
    audio_features_array = np.array([audio_features_array])
    #scale the audio features
    scaler = joblib.load('scaler.joblib')
    scaler.transform(audio_features_array)

    # lyrics_array = sl.get_lyrics_from_web('Rap God', 'Eminem')
    lyrics_array = sl.apiLyrics(title, artist)
    lyrics_array = np.array([lyrics_array])

    # print(lyrics_array.shape)
    print(audio_features_array.shape)
    print(lyrics_array)
    predictedGenre = modelPrediction(audio_features_array, lyrics_array)

    #print the predicted genre using format
    print(f"The predicted genre is: {predictedGenre}")

    return predictedGenre


# @app.get("/")
# def fetchData(title, artist):
#     return {"genre": findGenre(title, artist)}

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/submit")
async def handle_form(title: str = Form(...), artist: str = Form(...)):
    searchtxt = title + ' ' + artist
    url = findSongURL(searchtxt)
    filetxt = artist + '-' + title
    downloadSongfromYT(url, filetxt)
    genre = findGenre(title, artist)  # Assuming findGenre is defined
    return {"Title": title, "Artist": artist, "Predicted Genre": genre}



def findSongURL(query):
    # Set up the API key and build the service
    api_key = 'AIzaSyBsyeEg9-bJa16_j5nXgGfmn2iB_Vcwk9M'  # Replace with your API key
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Perform the search
    request = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=1
    )
    response = request.execute()

    # Extract the video URL
    if response['items']:
        video_id = response['items'][0]['id']['videoId']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        return video_url
    else:
        return "No results found"


def downloadSongfromYT(url, filetxt):
    # url input from user
    yt = YouTube(
        str(url))


    # extract only audio
    video = yt.streams.filter(only_audio=True).first()

    # check for destination to save file
    currentPath = os.getcwd()
    # in current path go to Datasets/2018655052_small
    destination = os.path.join(currentPath, 'Tests')

    # download the file
    out_file = video.download(output_path=destination)

    # save the file
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    # # rename as filetxt.mp3
    # write new_file = filetxt + '.mp3' at destination
    new_file = os.path.join(destination, filetxt + '.mp3')

    #
    os.rename(out_file, new_file)

    # result of success
    print(yt.title + " has been successfully downloaded.")
