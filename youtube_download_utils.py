import os
import yaml
import yt_dlp



def download_youtube_audio(url: str,
                           output_dir: str,
                           filename: str | None) -> str:
    """
    Download audio from a YouTube video and save it as a WAV file.

    Parameters
    ----------
    url : str
        The URL of the YouTube video to download.
    output_dir : str
        Directory where the WAV audio file is saved.
    filename : str | None
        Optional custom filename (without extension) for the saved audio.
        If None, the video ID will be used as the filename.

    Returns
    -------
    str
        Full path to the downloaded WAV audio file.
    """
    # Create the save directory if it doesn't already exist.
    os.makedirs(output_dir,
                exist_ok=True)

    # yt-dlp options to download best audio and convert it to WAV format.
    ydl_opts = {
        # Download best available audio quality.
        "format": "bestaudio/best",
        # Output template (uses video ID as filename).
        "outtmpl": os.path.join(output_dir, '%(id)s.%(ext)s'),
        "postprocessors": [{
            # Convert to audio after download.
            "key": "FFmpegExtractAudio",
            # Convert audio to WAV format.
            "preferredcodec": "wav",
            # Quality setting for conversion (not used for WAV but required).
            "preferredquality": "192",
        }],
        # Suppress yt-dlp console output.
        "quiet": True,
        # Suppress warnings (note: typo 'no_warningsz' should be 'no_warnings').
        "no_warningsz": True,
    }

    # Use yt-dlp to download and extract audio from the video URL
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        # Extract video ID to use as filename.
        video_id = info_dict.get("id", None) # type: ignore
        # Extract video title for logging.
        title = info_dict.get("title", "video") # type: ignore
        print(f"Downloaded: {title}")

    # Construct the default WAV filename based on video ID.
    wav_filename = f"{video_id}.wav"
    wav_path = os.path.join(output_dir, wav_filename)

    # If a custom filename is provided, rename the downloaded file accordingly.
    if filename:
        new_path = os.path.join(output_dir, filename + ".wav")
        os.rename(wav_path, new_path)
        wav_path = new_path

    # Return the final path to the saved WAV audio file
    return wav_path



if __name__ == "__main__":

    # Read the config file.
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Loop through all YouTube URLs specified in the config and download their audio.
    for url in config["youtube_data_url_list"]:
        path = download_youtube_audio(url=url,
                                      output_dir=config["dataset_save_location"],
                                      filename=None)
        
        print(f"Audio saved to: {path}")
