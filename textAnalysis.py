from threading import Thread, Lock

from lyricsgenius import Genius
from requests.exceptions import Timeout
import json
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer




lock = Lock()




def get_lyrics_of_tracks(chunk_df, allTracks_df, genius):
    #temporary df as a copy of allTracks_df
    temp_df = allTracks_df.copy()
    for index, row in chunk_df.iterrows():
        track_id = index
        track_title = row['track_title']
        artist_name = row['artist_name']

        lyrics = ""
        try:

            song = genius.search_song(track_title, artist_name)
            if song is not None:
                lyrics = song.lyrics
            else:
                lyrics = ""
        except Timeout:
            lyrics = ""
            print("Timeout")

        #handle AssertionError: Response status code was neither 200, nor 204! It was 502
        except AssertionError:
            lyrics = ""
            print("AssertionError")

        #handle AttributeError: 'float' object has no attribute 'translate'
        except AttributeError:
            lyrics = ""
            print("AttributeError")

        #update lyrics to temporary df
        temp_df.at[track_id, 'trackLyrics'] = lyrics

    #with lock replace data in allTracks_df with temporary df
    with lock:
        allTracks_df.update(temp_df)

    # first line of the string is the title of the song
    # print(lyrics.split("\n")[0])

    # #remove the first line of the string
    # lyrics = "\n".join(lyrics.split("\n")[1:])
    #
    # #remove words in brackets
    # lyrics = re.sub(r'\[.*?\]', '', lyrics)
    #
    # #remove words in parentheses
    # lyrics = re.sub(r'\(.*?\)', '', lyrics)

    # print(lyrics)


def getLyricsusingAPI(chunk_df, allTracks_df, API):
    # temporary df as a copy of allTracks_df
    temp_df = allTracks_df.copy()
    for index, row in chunk_df.iterrows():
        track_id = index
        track_title = row['track_title']
        artist_name = row['artist_name']

        API.artist = artist_name
        API.title = track_title

        API.getLyrics(save=False, ext='lrc')

        lyrics = API.lyrics

        # update lyrics to temporary df
        temp_df.at[track_id, 'trackLyrics'] = lyrics

    # with lock replace data in allTracks_df with temporary df
    with lock:
        allTracks_df.update(temp_df)


def getAllLyrics(allTracks_df):
    token = "f4qYxwv3xVqL6zWvKj5Xp8x0Z5Q5q4k4U7Z6w8D5V7k5S2uA9Q"
    genius = Genius(token, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], timeout=60,
                    retries=2, verbose=False, remove_section_headers=True)

    threads = []
    chunk_size = 100  # Number of songs per thread

    for i in range(0, len(allTracks_df), chunk_size):
        chunk_df = allTracks_df.iloc[i:i + chunk_size]
        t = Thread(target=get_lyrics_of_tracks, args=(chunk_df, allTracks_df, genius))
        t.start()
        threads.append(t)

    # for leftover tracks if allTracks_df is not divisible by chunk_size
    if len(allTracks_df) % chunk_size != 0:
        chunk_df = allTracks_df.iloc[len(allTracks_df) - (len(allTracks_df) % chunk_size):]
        t = Thread(target=get_lyrics_of_tracks, args=(chunk_df, allTracks_df, genius))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    return allTracks_df

def getAllAPIlyrics(allTracks_df):
    import azapi
    API = azapi.AZlyrics('google', accuracy=0.5)
    threads = []
    chunk_size = 100  # Number of songs per thread

    for i in range(0, len(allTracks_df), chunk_size):
        chunk_df = allTracks_df.iloc[i:i + chunk_size]
        t = Thread(target=getLyricsusingAPI, args=(chunk_df, allTracks_df, API))
        t.start()
        threads.append(t)

    # for leftover tracks if allTracks_df is not divisible by chunk_size
    if len(allTracks_df) % chunk_size != 0:
        chunk_df = allTracks_df.iloc[len(allTracks_df) - (len(allTracks_df) % chunk_size):]
        t = Thread(target=getLyricsusingAPI, args=(chunk_df, allTracks_df, API))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    return allTracks_df



def lyrics_for_eachSong(filtered_tracks):
    #dataframe to store track_id as index. trackLyrics and genre_top as feature columns
    # lyrics_df = pd.DataFrame(index=filtered_tracks.index, columns=['artist_name', 'track_title', 'trackLyrics', 'genre_top'])

    #for each track from tracks dataframe get the track_id, (artist, name) and (track, title)
    # for index, row in filtered_tracks.iterrows():
        #update artist_name and track_title to lyrics_df
        # lyrics_df.loc[index, 'artist_name'] = row['artist', 'name']
        # lyrics_df.loc[index, 'track_title'] = row['track', 'title']
        # lyrics_df.loc[index, 'genre_top'] = row['track', 'genre_top']

    lyrics_df = filtered_tracks[['artist_name', 'track_title', 'trackLyrics', 'genre_top']].copy()

    # allLyrics_df = getAllLyrics(lyrics_df)
    allLyrics_df = getAllAPIlyrics(lyrics_df)
    print(allLyrics_df.head(5))
    print(allLyrics_df.shape)

    #save the dataframe to csv file for future use
    allLyrics_df.to_csv('allAPILyrics.csv')


def main():
    filtered_Lyrics_df = pd.read_csv('filtered_lyrics.csv', index_col=0)
    lyrics_for_eachSong(filtered_Lyrics_df)


if __name__ == '__main__':
    main()