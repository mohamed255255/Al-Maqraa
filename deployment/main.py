import base64
import os
from typing import List
import librosa
import numpy as np
import torch
import nemo.collections.asr as nemo_asr
from omegaconf import DictConfig
from ruamel.yaml import YAML
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel



app = FastAPI()


config_path = './configs/config.yaml'
yaml = YAML(typ='safe')

with open(config_path, 'r', encoding='utf-8') as f:
    params = yaml.load(f)


model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']))


checkpoint_path = './checkpoint/best-checkpoint-epoch=131-val_loss=39.20.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()


class AudioInput(BaseModel):
    audio: str


def merge_segments(intervals, min_silence_duration=0.5, sample_rate=None):
    merged_intervals = []
    prev_end = None
    for start, end in intervals:
        if prev_end is None:
            prev_end = end
            merged_intervals.append((start, end))
        else:
            silence_duration = (start - prev_end) / sample_rate
            if silence_duration < min_silence_duration:
                merged_intervals[-1] = (merged_intervals[-1][0], end)
            else:
                merged_intervals.append((start, end))
            prev_end = end
    return merged_intervals


def remove_silence(audio, sample_rate, top_db=30, min_silence_duration=0.5):
    non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
    merged_intervals = merge_segments(non_silent_intervals, min_silence_duration, sample_rate)
    segments = [audio[start:end] for start, end in merged_intervals]
    processed_audio = np.concatenate(segments)
    return processed_audio.astype(np.float32)


def spectral_subtraction(audio_data, alpha=2.0):
    stft_matrix = librosa.stft(audio_data)
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    noise_spectrum = np.median(magnitude, axis=1)
    clean_magnitude = np.maximum(magnitude - alpha * noise_spectrum[:, np.newaxis], 0)
    clean_stft = clean_magnitude * np.exp(1j * phase)
    clean_audio = librosa.istft(clean_stft)
    return clean_audio


def preemphasis(signal, preemphasis_coeff=0.97):
    return np.append(signal[0], signal[1:] - preemphasis_coeff * signal[:-1])


def preprocessing(temp_wav, sample_rate):
    cleaned_audio = spectral_subtraction(temp_wav)
    cleaned_audio = preemphasis(cleaned_audio)
    cleaned_audio = remove_silence(cleaned_audio, sample_rate)
    return cleaned_audio


def transcribe_audio(temp_wav_path):
    try:
        transcription = model.transcribe(paths2audio_files=[temp_wav_path], batch_size=1)
        return transcription[0]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "Error during transcription"





@app.post('/transcribe')
async def transcribe_endpoint(audio_data: AudioInput):
    try:
        audio_bytes = base64.b64decode(audio_data.audio)

        temp_wav_path = './temp.wav'
        with open(temp_wav_path, 'wb') as f:
            f.write(audio_bytes)

        transcription_result = transcribe_audio(temp_wav_path)

        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

        return {"transcription": transcription_result}
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the audio")




if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=80)
