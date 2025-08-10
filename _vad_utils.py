import numpy as np
import os
from pydub import AudioSegment
import torch
import torch.nn as nn
from typing import List, Tuple
from typing import Callable, List, Dict
import yaml



def load_silero_vad():
    """
    
    """
    model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                                  model="silero_vad",
                                  force_reload=False,
                                  onnx=False)  # type: ignore

    (get_speech_timestamps, _, _, _, _) = utils
    return model, get_speech_timestamps


def load_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
    """
    Loads a WAV audio file, ensures it is mono and has a sampling
    rate of 16kHz, and returns the audio as a normalized PyTorch tensor.

    Parameters
    ----------
    audio_path : str)
        Path to the input WAV audio file.

    Returns
    -------
    Tuple[torch.Tensor, int]
    A tuple containing:
        - A 1D PyTorch tensor of normalized audio samples 
        (float32 in range [-1.0, 1.0]).
        - The sample rate (int), which will always be 16000.
    """
    # Load audio file using pydub.
    audio = AudioSegment.from_wav(audio_path)

    # Convert to mono if the audio has more than one channel.
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Resample to 16 kHz if the sample rate is different.
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)

    # Extract raw audio samples and convert to NumPy float32 array.
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    # Normalize sample values to range [-1.0, 1.0].
    samples /= 32768.0

    # Convert the NumPy array to a contiguous PyTorch tensor.
    wav_tensor = torch.from_numpy(samples).contiguous()

    return wav_tensor, 16000


def split_long_interval(start: float,
                        end: float,
                        speech_timestamps: List[dict],
                        target_length: float,
                        min_silence_for_split: float,
                        force_split: bool) -> List[Tuple[float, float]]:
    """
    Split a long speech interval [start, end] into smaller chunks based
    on silences and desired chunk length.

    The function uses detected speech timestamps to find silences within
    the interval, splitting at silences longer than `min_silence_for_split`
    seconds if the current chunk has reached the `target_length`. If `force_split`
    is True and the chunk length exceeds the target length, the function splits
    regardless of silence duration. If no suitable silence is found for splitting
    but forced splitting is requested, the interval is split into fixed-size
    chunks of approximately `target_length`.

    Parameters
    ----------
    start : float
        Start time of the long interval (in seconds).
    end : float
        End time of the long interval (in seconds).
    speech_timestamps : List[dict]
        List of speech segment dictionaries with 'start' and 'end' keys in
        sample units. Sample rate is assumed to be 16000 Hz.
    target_length : float
        Desired maximum duration (in seconds) for each chunk.
    min_silence_for_split : float
        Minimum silence duration (in seconds) required between speech segments
        to allow a split.
    force_split : bool
        If True, forcibly split chunks longer than `target_length` even
        if no silence is found.

    Returns
    -------
    List[Tuple[float, float]]
        List of tuples representing the start and end times (in seconds)
        of the split chunks.
    """
    # Filter speech timestamps that lie completely inside the interval,
    # convert samples to seconds.
    inside = [ts for ts in speech_timestamps if ts["start"]/16000 >= start \
              and ts["end"]/16000 <= end]
    # Extract the boundaries of speech segments (start and end times in seconds)
    boundaries = [(ts["start"]/16000, ts["end"]/16000) for ts in inside]

    chunks = []
    chunk_start = start

    for i in range(len(boundaries)-1):
        current_end = boundaries[i][1]
        next_start = boundaries[i+1][0]
        silence = next_start - current_end
        chunk_duration = current_end - chunk_start

        # Decide to split if:
        # Silence between segments is at least min_silence_for_split AND chunk is long enough
        # OR
        # Forced splitting is enabled and chunk is long enough
        if (silence >= min_silence_for_split and chunk_duration >= target_length) \
            or (force_split and chunk_duration >= target_length):
            chunks.append((chunk_start, current_end))
            chunk_start = next_start

        # Add the final chunk from the last split point to the end of the interval.
    if chunk_start < end:
        chunks.append((chunk_start, end))

    # If no chunks were created but force_split is True and the interval
    # is longer than target_length, split interval into fixed-length chunks
    # regardless of silence.
    if len(chunks) == 0 and force_split and (end - start) > target_length:
        duration = end - start
        num = int(np.ceil(duration / target_length))
        chunks = []

        for i in range(num):
            s = start + i * target_length
            e = min(start + (i+1) * target_length, end)
            chunks.append((s, e))

    return chunks


def vad_chunk_audio(audio_path : str,
                    output_folder : str,
                    target_chunk_length_sec : float,
                    max_silence_for_merge : float,
                    min_silence_for_split : float,
                    force_split : bool,
                    model : nn.Module,
                    get_speech_timestamps: Callable[[torch.Tensor, nn.Module, int],
                                                    List[Dict]]):
    """
    Perform voice activity detection (VAD) on an audio file, segmenting
    it into chunks based on speech activity and silence, then save these
    chunks as separate WAV files.

    The function loads the audio, detects speech segments using VAD,
    merges nearby speech intervals separated by silences shorter than
    `max_silence_for_merge`, and splits longer intervals into smaller
    chunks according to `target_chunk_length_sec`. Splitting leverages
    detected silences of at least `min_silence_for_split` seconds, or
    forcibly splits if enabled.

    Parameters
    ----------
    audio_path : str
        Path to the input WAV audio file.
    output_folder : str
        Directory where the output audio chunks will be saved.
    target_chunk_length_sec : float, optional
        Desired maximum length in seconds for each output chunk
        (default is 15 seconds).
    max_silence_for_merge : float, optional
        Maximum silence duration in seconds to merge adjacent speech
        intervals (default 2.0s).
    min_silence_for_split : float, optional
        Minimum silence duration in seconds required to split long
        intervals (default 0.5s).
    force_split : bool, optional
        If True, forcibly split chunks longer than `target_chunk_length_sec`
        even if no sufficient silence is detected (default True).
    model : torch.nn.Module
        Silero VAD model loaded via torch.hub.
    get_speech_timestamps : Callable[[torch.Tensor, torch.nn.Module, int], List[Dict]]
        Function to detect speech timestamps from audio tensor using the model.

    Returns
    -------
    None

    Side Effects
    ------------
    Saves segmented audio chunks as WAV files in `output_folder`
    with filenames formatted as "chunk_0000.wav", "chunk_0001.wav", etc.
    """
    # Ensure the output directory exists.
    os.makedirs(output_folder, exist_ok=True)

    # Load audio waveform and sample rate using a helper function.
    wav, sample_rate = load_audio(audio_path)

    # Obtain speech timestamps using VAD model (assumed to be globally available).
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sample_rate)  # type: ignore

    # Exit if there is no speech detected.
    if not speech_timestamps:
        print("No speech detected.")
        return

    # Convert speech timestamps from sample units to seconds.
    speech_intervals = [(ts["start"]/sample_rate, ts["end"]/sample_rate) \
                        for ts in speech_timestamps]

    # Merge speech intervals that are separated by silences shorter than max_silence_for_merge.
    merged = []
    cur_start, cur_end = speech_intervals[0]

    for start, end in speech_intervals[1:]:
        if start - cur_end <= max_silence_for_merge:
            # Extend current interval to include this speech segment.
            cur_end = end

        else:
            # Save current merged interval and start a new one.
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    merged.append((cur_start, cur_end))

    # Split merged intervals longer than 1.5 times the target length into smaller chunks.
    final_chunks = []
    for start, end in merged:
        duration = end - start

        if duration <= target_chunk_length_sec * 1.5:
            # Interval short enough, keep as-is.
            final_chunks.append((start, end))

        else:
            # Split longer intervals into smaller chunks based on silence and forced splitting.
            splits = split_long_interval(start,
                                         end,
                                         speech_timestamps,
                                         target_chunk_length_sec,
                                         min_silence_for_split,
                                         force_split)
            final_chunks.extend(splits)

    # Load the full audio file using pydub for exporting chunks.
    full_audio = AudioSegment.from_wav(audio_path)

    # Get the audio file name for naming each chunk.
    original_audio_filename = os.path.splitext(os.path.basename(audio_path))[0]

    # Export each final chunk as a separate WAV file.
    for i, (start_sec, end_sec) in enumerate(final_chunks):

        # Convert seconds to milliseconds for pydub
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        chunk_audio = full_audio[start_ms:end_ms]

        filename = f"{original_audio_filename}_chunk_{i:04d}.wav"
        out_path = os.path.join(output_folder, filename)
        chunk_audio.export(out_path, format="wav")
        print(f"Saved chunk {i} from {start_sec:.2f}s to {end_sec:.2f}s --> {filename}")

    print(f"Total chunks saved: {len(final_chunks)}")



if __name__ == "__main__":
    # Load the Silero VAD model.
    model, get_speech_timestamps = load_silero_vad()

    # Read the connfig file.
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

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
