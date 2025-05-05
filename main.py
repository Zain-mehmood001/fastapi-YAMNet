from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import scipy
from scipy.io import wavfile
import io

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this to your domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load class names
def class_names_from_csv(class_map_csv_text):
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        return JSONResponse(status_code=400, content={"error": "Only .wav files are supported"})

    wav_bytes = await file.read()
    file_like = io.BytesIO(wav_bytes)

    try:
        sample_rate, wav_data = wavfile.read(file_like)
        sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
        waveform = wav_data / tf.int16.max
        scores, embeddings, spectrogram = model(waveform)
        scores_np = scores.numpy()
        mean_scores = scores_np.mean(axis=0)
        class_number = mean_scores.argmax()
        inferred_class = class_names[mean_scores.argmax()]
        confidence = float(mean_scores.max())

        return {
            "predicted_class": inferred_class,
            "class_number": class_number,
            "confidence": confidence
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
