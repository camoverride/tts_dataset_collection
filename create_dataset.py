import os
import yaml
from youtube_download_utils import download_youtube_audio
from vad_utils import vad_chunk_audio, load_silero_vad
from metadata_utils import transcribe_wavs_to_txt



# Read the config file.
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


# (1) Loop through all YouTube URLs specified in the config and download their audio.
for url in config["youtube_data_url_list"]:
    path = download_youtube_audio(url=url,
                                  output_dir=config["dataset_save_location"],
                                  filename=None)
    
    print(f"Audio saved to: {path}")


# (2) Chunk the audio files using VAD.

# Load the Silero VAD model.
model, get_speech_timestamps = load_silero_vad()

# Get the paths to every audio file.
audio_paths = [os.path.join(config["dataset_save_location"], f) \
                for f in os.listdir(config["dataset_save_location"]) \
                if f.lower().endswith(".wav")]

# Iterate over every audio file, chunking them and writing them to `chunked_wav_save_location`
for audio_path in audio_paths:

    vad_chunk_audio(audio_path=audio_path,
                    output_folder=config["chunked_wav_save_location"],
                    target_chunk_length_sec=5,
                    max_silence_for_merge=1.0,
                    min_silence_for_split=0.3,
                    force_split=True,
                    model=model,
                    get_speech_timestamps=get_speech_timestamps)
    

# (3) Construct a metadata file.

transcribe_wavs_to_txt(audio_dir=config["chunked_wav_save_location"],
                        output_txt=os.path.join(config["dataset_save_location"],
                                                config["metadata_filename"]),
                        model_name="base")
