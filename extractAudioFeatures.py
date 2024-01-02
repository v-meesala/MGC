import librosa
import numpy as np
from librosa import feature
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import os
from scipy.stats import skew, kurtosis
import utils


def getAudioFts(filename):
    print(filename)
    x, sr = librosa.load(filename, sr=None, mono=True)

    print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))

    #get audio feature names from the columns of final_audio.csv file
    columnNames = pd.read_csv('final_audio.csv', nrows=1).columns.tolist()

    # features names are column names except 'genre_top' and 'track_id'
    featureNames = columnNames[1:-1]

    # Define the window and hop length
    frame_length = 2048
    hop_length = 512

    # Extract features
    chroma_stft= librosa.feature.chroma_stft(y=x, sr=sr, n_fft=frame_length, hop_length=hop_length)
    rmse= librosa.feature.rms(y=x, frame_length=frame_length, hop_length=hop_length)
    spectral_centroid= librosa.feature.spectral_centroid(y=x, sr=sr, n_fft=frame_length, hop_length=hop_length)
    spectral_bandwidth= librosa.feature.spectral_bandwidth(y=x, sr=sr, n_fft=frame_length, hop_length=hop_length)
    zcr= librosa.feature.zero_crossing_rate(y=x, hop_length=hop_length)
    spectral_contrast= librosa.feature.spectral_contrast(y=x, sr=sr, n_fft=frame_length, hop_length=hop_length)
    spectral_rolloff= librosa.feature.spectral_rolloff(y=x, sr=sr, n_fft=frame_length, hop_length=hop_length)
    chroma_cens= librosa.feature.chroma_cens(y=x, sr=sr,  hop_length=hop_length)
    chroma_cqt= librosa.feature.chroma_cqt(y=x, sr=sr,  hop_length=hop_length)
    tonnetz= librosa.feature.tonnetz(y=x, sr=sr,  hop_length=hop_length)
    # Calculate MFCCs
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_fft=frame_length, hop_length=hop_length)

    spectFeatures = [chroma_stft, rmse, spectral_centroid, spectral_bandwidth, zcr, spectral_contrast, spectral_rolloff, chroma_cens, chroma_cqt, tonnetz, mfcc]
    spectFeaturesnames = ['chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'zcr', 'spectral_contrast', 'spectral_rolloff', 'chroma_cens', 'chroma_cqt', 'tonnetz', 'mfcc']
    audioFeatureDict = {}
    for i in range(len(spectFeatures)):
        feature = spectFeatures[i]
        #convert feature variable name to string
        featString = spectFeaturesnames[i]
        audioFeatureDict[featString + '_mean'] = np.mean(feature, axis=1)
        audioFeatureDict[featString + '_median'] = np.median(feature, axis=1)
        audioFeatureDict[featString + '_std'] = np.std(feature, axis=1)
        audioFeatureDict[featString + '_min'] = np.min(feature, axis=1)
        audioFeatureDict[featString + '_max'] = np.max(feature, axis=1)
        audioFeatureDict[featString + '_skew'] = skew(feature, axis=1)
        audioFeatureDict[featString + '_kurtosis'] = kurtosis(feature, axis=1)


    finalAudioFeat_df = {}
    missing = []
    #for each feature in featureNames, search in the audioFeatureDict and if not found add it a new list, otherwise add it to audioFeat_df
    for feature in featureNames:
        columnStr = str(feature)
        #last 3 characters of the columnStr is _03, separate it from the columnStr
        featStr = columnStr[:-3]
        if featStr in audioFeatureDict.keys():
            dim = int(columnStr[-2:])-1
            try:
                finalAudioFeat_df[columnStr] = audioFeatureDict[featStr][dim]
            except:
                # finalAudioFeat_df[columnStr] = audioFeatureDict[featStr]
                print("Error: " + columnStr)
                print(audioFeatureDict[featStr])
                missing.append(feature)
        else:
            missing.append(feature)

    print(len(featureNames))
    print("Missing features: ")
    print(missing)
    print(finalAudioFeat_df)


    # Initialize Spotify API client
    # client_id = '7ee1c1a9a4df4588aee40854b9fc7ae9'
    # client_secret = '05c407c916e84c19b4790deddc91de91'
    # client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    # sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    #convert to dataframe
    audio_features_array = np.array(list(finalAudioFeat_df.values()))

    return audio_features_array


