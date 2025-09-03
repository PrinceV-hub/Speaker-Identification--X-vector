
# XVector Speaker Recognition

A complete deep learning pipeline implementing X-Vector architecture for speaker recognition, verification, and identification using TensorFlow/Keras.

## Features
- MFCC feature extraction with cepstral mean normalization
- TDNN-based X-Vector neural network
- Speaker classification and embedding extraction
- Training, validation, and test dataset splitting
- Model evaluation with accuracy, confusion matrix, and EER
- Visualization with t-SNE and PCA
- Speaker verification using cosine similarity

## Installation

Required libraries:
- TensorFlow
- NumPy
- Librosa
- Matplotlib
- Seaborn
- Scikit-learn
- Pandas

## Usage

1. Load dataset and extract features
2. Build and train X-Vector model
3. Extract speaker embeddings
4. Perform speaker identification and verification

## Example

```python
from speaker_recognition import predict_speaker

speaker_prototypes = load_speaker_prototypes('speaker_prototypes.pkl')
audio_file = 'sample.wav'
predict_speaker(audio_file, speaker_prototypes)
```

## License
MIT License
