import os
from audio_utils import normalize_audio
import torchaudio
import soundfile as sf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing the audio files to normalize')
args = parser.parse_args()

args.folder = "./synth_output-multilingual-matcha-vocos"
files = [file for file in os.listdir(args.folder) if file.endswith(".wav")]

count = 0
for file in files:
    try:
        yhat, sr = torchaudio.load(f"{args.folder}/{file}")
        yhat_normalized = normalize_audio(yhat, sample_rate=sr)
        yhat_normalized = yhat_normalized.numpy()
        yhat_normalized = yhat_normalized.transpose()
        sf.write(f'{args.folder}/normalized/{file}', yhat_normalized, sr, subtype='PCM_24')
    except Exception as err:
        print(f"failed to normalize {file}: {err}")
        count += 1

print(f"total failures: {count}")
