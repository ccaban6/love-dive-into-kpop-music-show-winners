import os
from pydub import AudioSegment

def convert_wav_to_flac_in_directory(directory):
    """
    Converts all WAV files in the specified directory to FLAC format
    and deletes the original WAV files.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                flac_path = wav_path.replace(".wav", ".flac")

                # Convert WAV to FLAC
                audio = AudioSegment.from_file(wav_path, format="wav")
                audio.export(flac_path, format="flac")

                # Delete the original WAV file
                os.remove(wav_path)

if __name__ == "__main__":
    audio_directory = "data/audio"
    convert_wav_to_flac_in_directory(audio_directory)