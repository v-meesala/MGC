import os

import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import sklearn
import sklearn.decomposition as skldecomp
import sklearn.preprocessing as sklpreprop

import librosa
import librosa.display, librosa.feature
from keras import Sequential
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, SimpleRNN, LSTM, Dropout, Bidirectional
from keras.src.preprocessing.text import Tokenizer
from keras.src.regularizers import l2
from keras.src.utils import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.utils import compute_class_weight

from imblearn.over_sampling import SMOTE, RandomOverSampler
from keras.utils import to_categorical

import textAnalysis
import trainingModels as tms
from gensim.models import Word2Vec
from transformers import BertTokenizer, TFBertForSequenceClassification

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

import re

import utils
import ast

def main():
    # AUDIO_DIR = os.environ.get('Datasets/2018655052_small')
    # get current project path
    currentPath = os.getcwd()
    # # in current path go to Datasets/2018655052_small
    # filepath = os.path.join(currentPath, 'Datasets\\fma_metadata\\tracks.csv')
    #
    # # Load metadata and features.
    tracks = load(os.path.join(currentPath, 'Datasets\\fma_metadata\\tracks.csv'))
    genres = load(os.path.join(currentPath, 'Datasets\\fma_metadata\\genres.csv'))
    features = load(os.path.join(currentPath, 'Datasets\\fma_metadata\\features.csv'))
    echonest = load(os.path.join(currentPath, 'Datasets\\fma_metadata\\echonest.csv'))
    #
    # np.testing.assert_array_equal(features.index, tracks.index)
    # assert echonest.index.isin(tracks.index).all()
    #
    # print(tracks.shape, genres.shape, features.shape, echonest.shape)
    #
    # allFeatureNames = features.columns.tolist()
    #
    # small = tracks[tracks['set', 'subset'] <= 'small']
    #
    # print('{} top-level genres'.format(len(genres['top_level'].unique())))
    # # ipd.display(genres.loc[genres['top_level'].unique()].sort_values('#tracks', ascending=False))
    #
    #
    #
    #
    # filtered_tracks = tracks[tracks.index.isin(echonest.index)]
    # # remove tracks with genre as NaN
    # filtered_tracks = filtered_tracks.dropna(subset=[('track', 'genre_top')])
    # # Remove rows with 'genre_top' as blank
    # filtered_tracks = filtered_tracks[filtered_tracks[('track', 'genre_top')] != '']
    #
    # #save the dataframe to csv file for future use
    # filtered_tracks.to_csv('filtered_tracks.csv')

    # #read from csv file and store in dataframe
    # filtered_tracks_df = pd.read_csv('filtered_tracks.csv', index_col=0, header=[0, 1])
    # #
    # textAnalysis.lyrics_for_eachSong(filtered_tracks_df)

    # read from csv file and store in dataframe
    allLyrics_df = pd.read_csv('allAPILyrics.csv', index_col=0)
    #
    # # #remove rows with trackLyrics as "" or NaN
    allLyrics_df = allLyrics_df.dropna(subset=['trackLyrics'])
    allLyrics_df = allLyrics_df[allLyrics_df['trackLyrics'] != '']

    print(allLyrics_df.shape)

    #remove rows with dash_count > 3
    allLyrics_df['dash_count'] = [str(lyric).count('" -') for lyric in allLyrics_df['trackLyrics']]
    allLyrics_df = allLyrics_df[allLyrics_df.dash_count < 3]

    # allLyrics_df['verse_count'] = [str(lyric).count('[Verse') for lyric in allLyrics_df['trackLyrics']]

    #
    # # #loop through each row of the dataframe
    for index, row in allLyrics_df.iterrows():
        #get the lyrics for each song
        lyrics = row['trackLyrics']
        # replace r'^.*?Lyrics' with ''
        lyrics = re.sub(r'^.*?Lyrics', '', lyrics)
        # replace '\n' with ' ' (space)
        lyrics = re.sub(r'\n', ' ', lyrics)
        #remove words in brackets
        lyrics = re.sub(r'\[.*?\]', '', lyrics)
        #remove words in parentheses
        lyrics = re.sub(r'\(.*?\)', '', lyrics)

        # stop words removal
        lyrics = clean_lyrics(lyrics)

        #update the lyrics for each song in the dataframe
        allLyrics_df.loc[index, 'trackLyrics'] = lyrics




    #create and add a column track_id which is the index of the dataframe
    allLyrics_df['tid'] = allLyrics_df.index

    excl = allLyrics_df[(allLyrics_df['trackLyrics'].str.contains('12.')) ]
    excl_ids = [i for i in excl['tid']]
    allLyrics_df = allLyrics_df[allLyrics_df['tid'].isin(excl_ids) == False]

    # print(allLyrics_df.head(5))

    #
    # #save the cleane lyrics dataframe to filtered_lyrics csv file for future use
    allLyrics_df.to_csv('filtered_lyrics.csv')

    #read from csv file and store in dataframe
    filtered_Lyrics_df = pd.read_csv('filtered_lyrics.csv', index_col=0)
    #
    # print(filtered_Lyrics_df.shape)
    # import audioAnalysis as aa
    # filtered_tracks_df = aa.audioFeatures(tracks, features, echonest)
    # print("result")
    # print(filtered_tracks_df.shape)
    filtered_tracks_df = pd.read_csv('final_audio.csv', index_col=0, header=[0, 1])
    #consider tarcks df rows which are in filtered_lyrics df only
    final_audio_df = filtered_tracks_df[filtered_tracks_df.index.isin(filtered_Lyrics_df.index)]
    print("final audio features")
    print(final_audio_df.shape)
    #save the dataframe to csv file for future use
    final_audio_df.to_csv('final_audio.csv')
    #
    # #consider lyrics df rows which are in filtered_tracks df only
    final_lyrics_df = filtered_Lyrics_df[['trackLyrics', 'genre_top']]
    print("final lyrics features")
    print(final_lyrics_df.shape)

    #save the dataframe to csv file for future use
    final_lyrics_df.to_csv('final_lyrics.csv')

    # textAnalysis.lyrics_for_eachSong(filtered_tracks_df)


    # #test the hybrid model
    import testingModels as tems

    tems.main()


def clean_lyrics(lyrics):
    # Tokenize the lyrics - split into words
    words = word_tokenize(lyrics)

    # Get English stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords
    filtered_lyrics = [word for word in words if word.lower() not in stop_words]

    # Optional: Remove punctuation and non-alphabetic characters
    filtered_lyrics = [word for word in filtered_lyrics if word.isalpha()]

    # Join words back into a string
    cleaned_lyrics = ' '.join(filtered_lyrics)

    return cleaned_lyrics



def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks






if __name__ == '__main__':
    main()