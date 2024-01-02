from lyricsgenius import Genius
from requests.exceptions import Timeout
import json
import pandas as pd
import re


# Dummy function: Replace this with your actual web scraping logic
def get_lyrics_from_web(track_title, artist_name):
    # Your web scraping code here
    lyrics = ""
    try:
        genius = Genius("f4qYxwv3xVqL6zWvKj5Xp8x0Z5Q5q4k4U7Z6w8D5V7k5S2uA9Q")
        song = genius.search_song(track_title, artist_name)
        lyrics = song.lyrics
    except Timeout:
        print("Timeout")
        return ""

    if lyrics == None:
        return ""

    # replace r'^.*?Lyrics' with ''
    lyrics = re.sub(r'^.*?Lyrics', '', lyrics)
    # replace '\n' with ' ' (space)
    lyrics = re.sub(r'\n', ' ', lyrics)
    # remove words in brackets
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    # remove words in parentheses
    lyrics = re.sub(r'\(.*?\)', '', lyrics)

    # Return lyrics as a string
    return lyrics


def apiLyrics(title, artist):
    import azapi

    API = azapi.AZlyrics('google', accuracy=0.5)

    API.artist = artist
    API.title = title

    API.getLyrics(save=False, ext='lrc')

    lyrics = API.lyrics

    # Correct Artist and Title are updated from webpage
    print(API.title, API.artist)

    return lyrics

