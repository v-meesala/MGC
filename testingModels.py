import keras
import numpy as np
from IPython.core.display import SVG
from imblearn.over_sampling import SMOTE, RandomOverSampler
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.src.layers import Embedding, Conv1D, GlobalMaxPooling1D, Add, concatenate, TextVectorization, \
    GlobalAveragePooling1D
from keras.src.optimizers import Adam
from keras.src.regularizers import l2
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences, plot_model, model_to_dot
import tensorflow as tf
import joblib


def testHybridModel(final_audio_df, final_lyrics_df, y):
    # Preprocessing
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_onehot = to_categorical(y_encoded)


    # Split the data
    X_audio_train, X_audio_test, X_lyrics_train, X_lyrics_test, y_train, y_test = train_test_split(
        final_audio_df, final_lyrics_df, y_onehot, test_size=0.2, random_state=42)

    #scaling on the audio features data
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(X_audio_train)
    scaler.transform(X_audio_test)

    #oversample the data
    X_audio_train, X_lyrics_train, y_train = oversample(X_audio_train, X_lyrics_train, y_train)

    # decode the one hot encoded y_train
    # y_train = label_encoder.inverse_transform(y_train.argmax(1))
    # #print the unique genre counts
    # print(pd.Series(y_train).value_counts())

    #save the fitted scaler
    joblib.dump(scaler, 'scaler.joblib')

    # Audio model
    input_audio = Input(shape=(X_audio_train.shape[1],))
    # x = Dense(X_audio_train.shape[1], activation='relu', kernel_regularizer=l2(0.001))(input_audio)
    # x = Dropout(0.7)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(input_audio)
    x = Dropout(0.7)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    audio_model = Model(inputs=input_audio, outputs=x)

    # Define TextVectorization layer
    max_tokens = 3000
    output_sequence_length = 100  # same as your maxlen
    text_vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length)
    batch_size = 100  # Choose an appropriate batch size
    for i in range(0, len(X_lyrics_train), batch_size):
        batch = X_lyrics_train['trackLyrics'].iloc[i:i + batch_size].values.astype(str)
        text_vectorizer.adapt(batch)

    # text_vectorizer.save('saved_text_vectorizer')

    # Lyrics model
    input_lyrics = Input(shape=(1,), dtype=tf.string)  # Input shape is set to 1 for raw text input
    x = text_vectorizer(input_lyrics)  # TextVectorization layer
    x = Embedding(input_dim=3000, output_dim=25)(x)
    x = Conv1D(filters=64, kernel_size=10, activation='relu')(x)
    #dropout
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling1D ()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    lyrics_model = Model(inputs=input_lyrics, outputs=x)

    # Concatenate
    concatenated = concatenate([audio_model.output, lyrics_model.output])

    # Output layer
    output = Dense(units=y_onehot.shape[1], activation='softmax')(concatenated)

    # Final model
    model = Model(inputs=[audio_model.input, lyrics_model.input], outputs=output)

    print(model.summary())
    # plot_model(model, to_file='hybrid_model.png', show_shapes=True, show_layer_names=True)
    # SVG(model_to_dot(model, show_shapes=True, show_layer_names=True, dpi=65).create(prog='dot', format='svg'))

    # Compile
    #add regularization
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


    # Train
    model.fit([X_audio_train,  X_lyrics_train], y_train, epochs=20, batch_size=16, validation_data=([X_audio_test, X_lyrics_test], y_test))

    # Evaluate
    # predicts = model.predict([X_audio_test, X_lyrics_test])

    # Assuming `label_encoder` is the fitted LabelEncoder instance
    joblib.dump(label_encoder, 'label_encoder.joblib')

    # When loading the model, include the TextVectorization layer in the custom_objects
    model.save('my_model.keras')


def main():
    # Load your data
    final_audio_df = pd.read_csv('final_audio.csv', low_memory=False, header=0)  # Replace with your actual DataFrame
    final_lyrics_df = pd.read_csv('final_lyrics.csv')  # Replace with your actual DataFrame

    #make track_id column as index and remove that column
    final_audio_df = final_audio_df.set_index('track_id')
    final_lyrics_df = final_lyrics_df.set_index('track_id')

    y = final_audio_df['genre_top']

    #remove the rows with genre_top as 'Experimental'
    final_audio_df = final_audio_df[final_audio_df['genre_top'] != 'Experimental']
    final_lyrics_df = final_lyrics_df[final_lyrics_df['genre_top'] != 'Experimental']
    y = y[y != 'Experimental']

    #change final_lyrics_df by removing genre_top column
    final_lyrics_df = final_lyrics_df.drop(columns=['genre_top'])

    #change final_audio_df by removing genre_top column
    final_audio_df = final_audio_df.drop(columns=['genre_top'])


    #print all shapes of dataframes
    print(final_audio_df.shape)
    print(final_lyrics_df.shape)
    print(y.shape)


    #print the unique genre counts
    print(y.value_counts())

    # Train the hybrid model
    testHybridModel(final_audio_df, final_lyrics_df, y)


def oversample(X_audio_train, X_lyrics_train, y_train):

    #copy lyrics df
    lyrics_df = X_lyrics_train.copy()

    #modify X_lyrics_train by removing trackLyrics column and replace it with index value as float
    X_lyrics_train = X_lyrics_train.drop(columns=['trackLyrics'])
    X_lyrics_train['track_id'] = X_lyrics_train.index.astype(float)

    #merge the dataframes
    merged_train = pd.concat([X_audio_train, X_lyrics_train], axis=1)

    #oversample the data using random oversampling
    ros = RandomOverSampler(random_state=42)
    merged_train, y_train = ros.fit_resample(merged_train, y_train)

    #split the data into audio and lyrics
    X_audio_train = merged_train.iloc[:, :X_audio_train.shape[1]]
    X_lyrics_train = merged_train.iloc[:, X_audio_train.shape[1]:]

    #update the track_id column float value in  X_lyrics_train to int index and get the corresponding lyrics from lyrics_df index
    X_lyrics_train['track_id'] = X_lyrics_train['track_id'].astype(int)
    X_lyrics_train['trackLyrics'] = lyrics_df.loc[X_lyrics_train['track_id'], 'trackLyrics'].values

    #remove the track_id column
    X_lyrics_train = X_lyrics_train.drop(columns=['track_id'])

    print(X_audio_train.shape)
    print(X_lyrics_train.shape)
    print(y_train.shape)

    return X_audio_train, X_lyrics_train, y_train


if __name__ == '__main__':
    main()