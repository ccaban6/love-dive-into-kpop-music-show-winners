import os
import yt_dlp
import pandas as pd
from datetime import datetime
import json

# Config

OUTPUT_DIR = "data/audio/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data/logs", exist_ok=True)

LOG_FILE = "data/logs/download_log.csv"

def download_song(song_name: str, artist: str, min_duration: int = 60, max_duration: int = 600):
    """
    Download audio from YouTube given a song name and artist.
    
    Args:
        song_name (str): Name of the song
        artist (str): Name of the artist
        min_duration (int): Minimum acceptable duration in seconds
        max_duration (int): Maximum acceptable duration in seconds
    
    Returns:
        dict: Metadata about the downloaded song (or None if failed)
    """

    query = f"ytsearch1:{artist} {song_name} official audio"

    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "extractaudio": True,
        "quiet": True,
        "default_search": "ytsearch1",  # Take first result only
        "outtmpl": os.path.join(OUTPUT_DIR, f"{artist} - {song_name}.%(ext)s"),
        "keepvideo": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": {
                 "ffmpeg": [
                    "-ar", "16000",  # Set sample rate to 16kHz
                    "-ac", "1",      # Convert to mono
                    "-acodec", "pcm_s16le"  # 16-bit PCM encoding
                ]
            }
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=True)
            with open("test.json", "w") as file:
                json.dump(info, file)
            duration = info.get("entries", {})[0].get("duration_string", "Not found")
            minutes, seconds = map(int, duration.split(":"))
            total_seconds = 60 * minutes + seconds

            title = info.get("entries", {})[0].get("fulltitle", "unknown")
            uploader = info.get("entries", {})[0].get("uploader", "unknown")
            webpage_url = info.get("entries", {})[0].get("original_url", "unknown")

            # Duration sanity check
            if not (min_duration <= total_seconds <= max_duration):
                print(f"❌ Skipping {artist} - {song_name}: invalid duration ({duration}s).")
                return None

            metadata = {
                "artist": artist,
                "song": song_name,
                "title": title,
                "uploader": uploader,
                "duration_sec": total_seconds,
                "url": webpage_url,
                "downloaded_at": datetime.now().isoformat(),
            }

            # Append metadata to log
            log_metadata(metadata)

            print(f"✅ Downloaded {artist} - {song_name} ({duration}s)")
            print(verify_audio_specs(os.path.join(OUTPUT_DIR, f"{artist} - {song_name}.wav")))
            return metadata

    except Exception as e:
        print(f"❌ Failed to download {artist} - {song_name}: {e}")
        return None
    
def log_metadata(metadata: dict):
    """
    Append metadata to a CSV log file.
    """
    df = pd.DataFrame([metadata])

    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

def verify_audio_specs(audio_file_path):
    """
    Verify that the audio file meets Essentia model requirements.
    """
    try:
        import librosa
        
        # Load audio file and check specifications
        y, sr = librosa.load(audio_file_path, sr=None)
        
        print(f"Audio file: {audio_file_path}")
        print(f"Sample rate: {sr} Hz")
        print(f"Duration: {len(y)/sr:.2f} seconds")
        print(f"Channels: {1 if y.ndim == 1 else y.shape[0]}")
        print(f"Data type: {y.dtype}")
        
        if sr == 16000:
            print("✓ Sample rate is correct for Essentia models")
        else:
            print("⚠ Sample rate should be 16000 Hz for most Essentia models")
            
        return sr == 16000
        
    except ImportError:
        print("Install librosa to verify audio specifications: pip install librosa")
        return True
    except Exception as e:
        print(f"Error verifying audio: {str(e)}")
        return False

if __name__ == "__main__":
    # Example test
    test_song = "World"
    test_artist = "SEVENTEEN"
    download_song(test_song, test_artist)