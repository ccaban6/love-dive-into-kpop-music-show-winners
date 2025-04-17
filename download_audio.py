import yt_dlp
import tempfile
import os
import urllib.parse
from pydub import AudioSegment

def get_permanent_audio_path(song_name, artist, ext="flac", base_dir="data/audio"):
    """
    Generates a permanent file path for the audio file.
    """
    # Clean the file name and build the full path.
    file_name = f"{song_name}_{artist}".replace(" ", "_").lower() + f".{ext}"
    return os.path.join(base_dir, file_name)

def convert_wav_to_flac(wav_path):
    """
    Converts a WAV file to FLAC format.
    """
    flac_path = wav_path.replace(".wav", ".flac")
    audio = AudioSegment.from_file(wav_path, format="wav")
    audio.export(flac_path, format="flac")
    return flac_path

def search_and_download_audio(song_name, artist, base_dir="data/audio"):
    """
    Searches YouTube for a song, downloads the audio, converts it to FLAC,
    and stores it permanently in base_dir.
    """
    # Ensure the permanent storage directory exists.
    os.makedirs(base_dir, exist_ok=True)

    # Generate the target file path in the permanent directory.
    target_path = get_permanent_audio_path(song_name, artist, ext="flac", base_dir=base_dir)

    # If the file is already present, skip download.
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        print(f"File already exists for {song_name} by {artist}. Skipping download.")
        return target_path

    query = f"{song_name} {artist} audio"
    search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"

    # Extract the first video URL using yt-dlp.
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "default_search": "ytsearch",
        "force_generic_extractor": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_results = ydl.extract_info(search_url, download=False)
            if not search_results or "entries" not in search_results or not search_results["entries"]:
                print(f"No results found for {query}")
                return None
            video_url = search_results["entries"][0]["url"]
    except Exception as e:
        print(f"Error searching for {query}: {e}")
        return None

    # Use a temporary directory for the intermediate MP3 download.
    temp_dir = tempfile.gettempdir()
    temp_output = os.path.join(temp_dir, f"{song_name}_{artist}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": temp_output,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Determine the temporary MP3 path.
        mp3_path = temp_output.replace("%(ext)s", "mp3")

        # Check if the downloaded file is empty.
        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
            print(f"Downloaded file is empty for {song_name} by {artist}")
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
            return None

        # Convert the MP3 to FLAC.
        audio = AudioSegment.from_file(mp3_path, format="mp3")
        audio.export(target_path, format="flac")

        # Clean up the temporary MP3 file.
        os.remove(mp3_path)

        print(f"Downloaded and saved {song_name} by {artist} to {target_path}")
        return target_path

    except Exception as e:
        print(f"Error processing {song_name} by {artist}: {e}")
        return None

def process_songs_dataframe(df, base_dir="data/audio"):
    """
    Processes a DataFrame with 'Song' and 'Artist' columns,
    ensuring downloads are done in order.
    """
    # Ensure the DataFrame is sorted by desired order (e.g., an index column)
    df = df.sort_values("Artist")
    df["file_path"] = df.apply(lambda row: search_and_download_audio(row["Song"], row["Artist"], base_dir), axis=1)
    return df







