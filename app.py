from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import numpy as np
import librosa
import soundfile as sf
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
BINAURAL_FOLDER = 'binauralwaves'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['BINAURAL_FOLDER'] = BINAURAL_FOLDER

# Load the model
with open('emotion_recognition_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Emotions dictionary
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Map emotions to corresponding audio files
emotion_to_audio = {
    'calm': 'alpha.mp3',
    'happy': 'beta.mp3',
    'fearful': 'theta.mp3',
    'disgust':  'theta.mp3',
    'sad':  'alpha.mp3',
    'surprised':  'beta.mp3',
    'neutral':  'beta.mp3',
    'disgust':  'alpha.mp3'
}

def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.wav'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.wav')
        file.save(file_path)
        feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
        feature = np.expand_dims(feature, axis=0)  # Add batch dimension
        prediction = model.predict(feature)
        emotion = prediction[0]

        if emotion in emotion_to_audio:
            audio_file = emotion_to_audio[emotion]
            audio_path = os.path.join(app.config['BINAURAL_FOLDER'], audio_file)
            return jsonify({'emotion': emotion, 'audio': audio_file})
        else:
            return jsonify({'emotion': emotion, 'audio': None})

    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/audio/<filename>')
def send_audio(filename):
    return send_from_directory(app.config['BINAURAL_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(BINAURAL_FOLDER):
        os.makedirs(BINAURAL_FOLDER)
    app.run(debug=True)