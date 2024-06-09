import librosa
import numpy as np
import os
from glob import glob
import soundfile as sf

def segment_audio(audio_file, save_dir, segment_duration=5):
    # Load the audio file
    audio, sample_rate = librosa.load(audio_file, sr=None)

    # Calculate the number of segments
    num_segments = int(np.floor(len(audio) / (sample_rate * segment_duration)))

    # Perform segmentation and save individual segments
    segment_paths = []
    for i in range(num_segments):
        start_sample = i * sample_rate * segment_duration
        end_sample = (i + 1) * sample_rate * segment_duration
        segment = audio[start_sample:end_sample]

        file_name = f"{os.path.splitext(os.path.basename(audio_file))[0]}_segment_{i}.wav"
        save_path = os.path.join(save_dir, file_name)

        sf.write(save_path, segment, sample_rate)
        segment_paths.append(save_path)

    return segment_paths

# Example usage
sample_dir = 'C:\\Users\\asus\\Masaüstü\\BitirmeProjesi\\AudioFiles\\'
save_directory = 'C:\\Users\\asus\\Masaüstü\\BitirmeProjesi\\segmnets'
audio_files = glob(sample_dir + '*.wav')
segment_duration = 5

for audio_file in audio_files:
    segment_paths = segment_audio(audio_file, save_directory, segment_duration)
    print(f"Segments saved for {audio_file}: {len(segment_paths)}")
