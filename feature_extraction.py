import librosa
import numpy as np

def extract_mfcc_features(audio_path, sr=16000, n_mfcc=23, n_fft=512, hop_length=160):
    """Extract MFCC features from audio file"""
    try:
        y, _ = librosa.load(audio_path, sr=sr)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, 
            hop_length=hop_length, window='hamming', center=False
        )
        
        mfccs = mfccs - np.mean(mfccs, axis=1, keepdims=True)
        return mfccs.T
        
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def create_segments(features, window_length=400, step_size=200):
    """Create fixed-length segments from features"""
    if features.shape[0] < window_length:
        padding = window_length - features.shape[0]
        features = np.pad(features, ((0, padding), (0, 0)), mode='constant')
        return np.array([features])
    
    segments = []
    start = 0
    
    while start + window_length <= features.shape[0]:
        segment = features[start:start + window_length]
        segments.append(segment)
        start += step_size
        
        if len(segments) >= 50:
            break
    
    return np.array(segments) if segments else np.array([])
