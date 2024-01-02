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
import spotipy
from sklearn.utils import compute_class_weight
from spotipy.oauth2 import SpotifyClientCredentials
from imblearn.over_sampling import SMOTE, RandomOverSampler
from keras.utils import to_categorical
import trainingModels as tms
from gensim.models import Word2Vec
from transformers import BertTokenizer, TFBertForSequenceClassification

import re

import utils



def audioFeatures(tracks, features, echonest):


    # small = tracks['set', 'subset'] <= 'small'
    # genre1 = tracks['track', 'genre_top'] == 'Instrumental'
    # genre2 = tracks['track', 'genre_top'] == 'Hip-Hop'
    # X = features.loc[small & (genre1 | genre2), 'mfcc']
    # X = skldecomp.PCA(n_components=2).fit_transform(X)
    # y = tracks.loc[small & (genre1 | genre2), ('track', 'genre_top')]
    # y = sklpreprop.LabelEncoder().fit_transform(y)
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', alpha=0.5)
    # plt.title('PCA Scatter Plot of Instrumental and Hip-Hop')
    # plt.show()
    # print(X.shape, y.shape)

    print(tracks.shape)
    print("........")
    filtered_tracks = tracks[tracks.index.isin(echonest.index)]
    print(filtered_tracks.shape)
    #remove tracks with genre as NaN
    filtered_tracks = filtered_tracks.dropna(subset=[('track', 'genre_top')])
    # Remove rows with 'genre_top' as blank
    filtered_tracks = filtered_tracks[filtered_tracks[('track', 'genre_top')] != '']
    # Remove duplicate rows based on index (or any other criteria)
    duplicates = filtered_tracks.index.duplicated(keep=False)
    print(duplicates)
    if any(duplicates):
        print("Duplicate index values found.")
    else:
        print("No duplicate index values found.")

    filtered_spectral_features = features[features.index.isin(filtered_tracks.index)]
    filtered_echonest_features = echonest[echonest.index.isin(filtered_tracks.index)]
    #consider only first 8 columns are filetred_echonest_features
    filtered_audio_features = filtered_echonest_features.iloc[:, :8]
    filtered_temporal_features = filtered_echonest_features.iloc[:, -224:]
    # filtered_spectral_features = filtered_spectral_features.loc[:, ['mfcc', 'spectral_contrast',
    #                                                 'chroma_cens', 'tonnetz','spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'rmse', 'zcr']]



    print(filtered_audio_features.shape)
    print(filtered_temporal_features.shape)
    print(filtered_spectral_features.shape)

    audio_features_names = filtered_audio_features.columns.tolist()
    temporal_features_names = filtered_temporal_features.columns.tolist()
    spectral_features_names = filtered_spectral_features.columns.tolist()



    X_1 =   filtered_spectral_features
    # X_1 = pd.concat([X_1, filtered_audio_features], axis=1)
    # X_1 = pd.concat([X_1, filtered_temporal_features], axis=1)
    # print(X_1.head(5))


    y_1 = filtered_tracks.loc[:, ('track', 'genre_top')]
    y_1 = y_1.cat.remove_unused_categories()
    # print(y_1)

    # Randomly split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.2, random_state=42)


    #find important features using XGBoost
    # top_temporal_features = XGBoost(X_train[temporal_features_names], y_train)
    # top_audio_features = XGBoost(X_train[audio_features_names], y_train)
    # top_spectral_features = XGBoost(X_train[spectral_features_names], y_train)

    x1size = X_train.shape
    xtrainsize = X_train.shape

    # selected_features = top_spectral_features + top_temporal_features + top_audio_features
    selected_features = XGBoost(X_train, y_train)



    #feature selection
    #consider only top features in X_train and X_test
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    newsize = X_train.shape

    ipd.display(y_train.value_counts())


    # '''Oversampling'''
    # smote = SMOTE(sampling_strategy='auto')  # Choose strategy based on your needs
    # X_train, y_train = smote.fit_resample(X_train, y_train)
    #random oversampling
    # ros = RandomOverSampler(random_state=42)
    # X_train, y_train = ros.fit_resample(X_train, y_train)
    # ipd.display(y_train.value_counts())

    print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
    print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))
    print(np.unique(y_train))




    # Initialize lists to store results
    accuracies = []
    feature_counts = []

    # # Iteratively add features and evaluate model
    # for i in range(1):
    #     #take subset of features from selected_features
    #     subset_features = selected_features
    #
    #     X_train_selected = X_train[ subset_features]
    #     X_test_selected = X_test[ subset_features]
    #
    #     # Be sure training samples are shuffled.
    #     X_train_selected, y_train = sklearn.utils.shuffle(X_train_selected, y_train, random_state=42)
    #     # # Standardize features by removing the mean and scaling to unit variance.
    #     scaler = StandardScaler(copy=False)
    #     scaler.fit_transform(X_train_selected)
    #     scaler.transform(X_test_selected)
    #
    #
    #     X_NN = X_1[subset_features]
    #     y_NN = y_1
    #     #train on NN
    #     acc = tms.trainOnNN(X_NN, y_NN)
    #
    #     accuracies.append(acc)
    #     feature_counts.append(i)
    #
    #
    #
    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.plot(feature_counts, accuracies, marker='o')
    # plt.xlabel('Number of Features')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs Number of Features')
    # plt.show()



    # tms.trainOnNN(X_1, y_1)

    # tms.trainOnClfs(X_train, y_train, X_test, y_test)
    #top 370 features as list data structure
    final_features = selected_features[:250]

    #save the X_1 dataframe with only final features as columns to csv file for future use
    final_tracks_df = X_1[final_features]
    #change the tuple column names to string column names
    final_tracks_df.columns = final_tracks_df.columns.map('_'.join)
    #add genre_top column to final_tracks_df from y_1 based on index from final_tracks_df
    final_tracks_df['genre_top'] = y_1[final_tracks_df.index]

    return final_tracks_df


def XGBoost(X_train, y_train):
    import xgboost as xgb
    from xgboost import plot_importance
    # Initialize XGBoost model
    model = xgb.XGBClassifier()

    #convert y_train to numeric encodings
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    # One-hot encode the labels
    y_onehot = to_categorical(y_encoded)

    # Fit the model to the training data
    model.fit(X_train, y_onehot)

    # Get feature importances
    feature_importance = model.feature_importances_

    #create an empty dataframe to store feature importances and feature names
    importance_df = pd.DataFrame(columns=['Feature', 'Importance'])

    feature_size = len(X_train.columns)
    # Save feature importances to a DataFrame importance_df
    for i, score in enumerate(feature_importance):
        importance_df.loc[i] = [X_train.columns[i], score]
        # print(f"Feature {i}: {score}")


    # Sort features by importance
    sorted_df = importance_df.sort_values(by='Importance', ascending=False)

    # print(sorted_df)

    #consider only top features with importance > 4.98e-03(temporal), 4.94e-03(spectral), 2.95e-03(All combined)
    # top_features = sorted_df[sorted_df['Importance'] > 2.95e-03]
    top_features = sorted_df
    print("Top features size : " + str(top_features.shape))


    # Plot feature importances
    # plot_importance(model)
    # plt.show()

    #return top features names as a list
    return top_features['Feature'].tolist()


def featureEnginering(filtered_spectral_features, filtered_temporal_features):

    '''Feature engineering'''
    fsf = filtered_spectral_features
    ftf = filtered_temporal_features
    #convert tuple columns to string columns
    fsf.columns = fsf.columns.map('_'.join)
    ftf.columns = ftf.columns.map('_'.join)
    #craete a new column tid and assign track_id to it for both dataframes
    fsf['tid'] = fsf.index
    ftf['tid'] = ftf.index

    print(ftf.columns)

    import featuretools as ft

    # Create a new entityset to hold our entities (tables)
    es = ft.EntitySet(id="music_features")

    # Add the dataframes to the entityset
    es = es.add_dataframe(
        dataframe_name="spectral_features",
        dataframe=fsf,
        index="track_id",
    )

    es = es.add_dataframe(
        dataframe_name="temporal_features",
        dataframe=ftf,
        index="track_id",
    )


    # Add the relationship to the entity set
    es = es.add_relationship("spectral_features", "track_id",
                                      "temporal_features", "tid")

    # relationship = ft.Relationship(es, parent_dataframe_name="spectral_features", parent_column_name="track_id",
    #                                child_dataframe_name="temporal_features", child_column_name="track_id")
    print(es)
    # # Add the relationship to the entity set
    # es = es.add_relationship(relationship)

    # Run deep feature synthesis
    features, feature_defs = ft.dfs(entityset=es,
                                    target_dataframe_name="spectral_features",
                                    verbose=True  # set to True to see progress
                                    )

    # features is a new dataframe with the engineered features
    print(feature_defs)
    return features, feature_defs
