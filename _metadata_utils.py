import os
import whisper
import re
import yaml



def normalize_text(text : str) -> str:
    """
    Normalize the text:
    - Uppercase.
    - Remove punctuation except apostrophes.
    - Collapse spaces.

    Parameters
    ---------
    text : str
        Transcribed audio text.
    
    Returns
    -------
    str
        Text normalized for Coqui TTS.    
    """
    text = text.upper()
    text = re.sub(r"[^A-Z0-9' ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def transcribe_wavs_to_txt(audio_dir : str,
                           output_txt : str,
                           model_name : str):
    """
    Transcribe all WAV audio files in a directory using a Whisper model, 
    normalize the transcriptions, and save the results to a text file
    formatted for Coqui TTS training.

    Each line in the output text file contains:
        <filename_without_extension>|<raw_transcription>|<normalized_transcription>

    Parameters
    ----------
    audio_dir : str
        Path to the directory containing input WAV files.
    output_txt : str
        Path to the output text file where transcriptions will be saved.
    model_name : str
        Name of the Whisper model to load (e.g., 'small', 'base', 'medium', etc.).

    Notes
    -----
    This function expects a helper function `normalize_text(text: str) -> str`
    to be defined in the same module, which converts raw transcriptions into
    normalized uppercase text suitable for Coqui TTS training.

    The transcription process uses the Whisper model's default transcription settings.
    """
    # Load the specified Whisper model
    model = whisper.load_model(model_name)

    # List all files in the directory that end with '.wav' (case-insensitive)
    wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]
    
    # Open the output text file.
    with open(output_txt, "w", encoding="utf-8") as f_out:

        # Process files in sorted order for consistency.
        for filename in sorted(wav_files):

            # Construct full file path for the audio file.
            audio_path = os.path.join(audio_dir, filename)

            # Transcribe the audio using Whisper.
            result = model.transcribe(audio_path)

            # Extract the raw text transcription and strip whitespace.
            text = result["text"].strip() # type: ignore

            # Normalize the text.
            norm_text = normalize_text(text)

            # Remove the '.wav' extension from the filename for output formatting.
            filename = os.path.splitext(filename)[0]
            
            # Format the output line: filename|raw_text|normalized_text.
            line = f"{filename}|{text}|{norm_text}\n"

            # Write the line to the output text file
            f_out.write(line)


            print(f"Processed {filename}")



if __name__ == "__main__":
    # Read the config file.
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    """
    Transcribe all the audio files to a file that looks like:

    file_0000|I am talking now,|I AM TALKING NOW

    Where the first column is the filename, the second is the text,
    and the third is the normalized text. It is separated by pipes |.
    """
    
    transcribe_wavs_to_txt(audio_dir=config["chunked_wav_save_location"],
                           output_txt=os.path.join(config["dataset_save_location"], "metadata.txt"),
                           model_name="base")
