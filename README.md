# Speaker Identification using X-Vector Embeddings

A deep learning-based speaker identification system using X-Vector embeddings implemented in TensorFlow/Keras. This system can identify speakers from audio recordings and extract speaker embeddings for verification tasks.

## Features

- **X-Vector Architecture**: Implementation of the standard X-Vector neural network for speaker recognition
- **MFCC Feature Extraction**: Robust audio feature extraction using Mel-Frequency Cepstral Coefficients
- **Speaker Identification**: Identify which speaker an audio sample belongs to
- **Speaker Verification**: Verify if an audio sample matches a claimed speaker identity
- **Embedding Extraction**: Extract 512-dimensional speaker embeddings for further analysis
- **Multi-Speaker Support**: Trained on 5 different speakers with high accuracy

## Dataset

This project uses the **Speaker Recognition Dataset** available on Kaggle:
- **Dataset Link**: [Speaker Recognition Dataset](https://www.kaggle.com/code/kongaevans/recognizing-a-speaker-with-spectrograms/input)
- **Contains**: 7,501 audio files from 5 speakers
- **Speakers**: Nelson Mandela, Benjamin Netanyahu, Margaret Thatcher, Jens Stoltenberg, Julia Gillard
- **Format**: 16kHz PCM WAV files, 1-second duration each

## Performance

- **Training Accuracy**: 98.40%
- **Validation Accuracy**: 93.49%
- **Test Accuracy**: 92.82%
- **Equal Error Rate (EER)**: ~8.5%
- **Embedding Dimension**: 512

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/speaker-identification-xvector.git
cd speaker-identification-xvector
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and extract to `data/` directory

## Quick Start

### 1. Training a New Model

```python
from src.train import train_xvector_model
from src.data_preprocessing import build_feature_dataset

# Load and preprocess data
X_train, y_train, label_encoder = build_feature_dataset("path/to/dataset")

# Train the model
model = train_xvector_model(X_train, y_train, num_classes=5)
```

### 2. Using Pre-trained Model for Prediction

```python
from src.inference import SpeakerIdentifier

# Initialize the identifier
identifier = SpeakerIdentifier(
    model_path="models/xvector_model.h5",
    prototypes_path="models/speaker_prototypes.pkl"
)

# Predict speaker from audio file
speaker, confidence = identifier.predict_speaker("path/to/audio.wav")
print(f"Predicted Speaker: {speaker} (Confidence: {confidence:.3f})")
```

### 3. Extract X-Vector Embeddings

```python
from src.inference import extract_xvector_embedding

# Extract 512-dimensional embedding
embedding = extract_xvector_embedding("path/to/audio.wav", model)
print(f"Embedding shape: {embedding.shape}")  # (512,)
```

## Project Structure

```
speaker-identification-xvector/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # MFCC extraction and segmentation
│   ├── model.py                 # X-Vector model architecture
│   ├── train.py                 # Training pipeline
│   ├── inference.py             # Prediction and embedding extraction
│   └── utils.py                 # Utility functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation_analysis.ipynb
├── models/
│   ├── xvector_model.h5         # Trained model (download separately)
│   └── speaker_prototypes.pkl   # Speaker prototypes
├── data/                        # Dataset directory (not included)
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

## Model Architecture

The X-Vector model consists of:

1. **Frame-level layers (TDNN)**: 5 Time-Delay Neural Network layers for processing audio frames
2. **Statistics Pooling**: Aggregates mean and standard deviation across time
3. **Segment-level layers**: 2 fully connected layers for speaker embedding
4. **Classification layer**: Softmax output for speaker identification

**Input**: (400, 23) - 400 MFCC frames with 23 coefficients each
**Output**: 512-dimensional speaker embedding + classification scores

## API Reference

### SpeakerIdentifier Class

```python
class SpeakerIdentifier:
    def __init__(self, model_path, prototypes_path)
    def predict_speaker(self, audio_path) -> Tuple[str, float]
    def verify_speaker(self, audio_path, claimed_speaker, threshold=0.8) -> Tuple[bool, float]
    def extract_embedding(self, audio_path) -> np.ndarray
```

### Key Functions

- `extract_mfcc_features(audio_path)`: Extract MFCC features from audio
- `create_segments(features)`: Create fixed-length segments for model input
- `get_xvector_embedding(audio_path)`: Extract speaker embedding
- `calculate_speaker_similarity(emb1, emb2)`: Compute cosine similarity between embeddings

## Training Details

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 64
- **Epochs**: 50 (with early stopping)
- **Data Split**: 77.2% train, 17.8% validation, 5% test
- **Augmentation**: Cepstral mean normalization per utterance

## Evaluation Metrics

- **Accuracy**: Classification accuracy on test set
- **Equal Error Rate (EER)**: Standard metric for speaker verification
- **Confusion Matrix**: Per-speaker performance analysis
- **Embedding Quality**: Intra-speaker vs inter-speaker distance analysis

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{xvector-speaker-identification,
  title={Speaker Identification using X-Vector Embeddings},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/speaker-identification-xvector}
}
```

## Acknowledgments

- Original X-Vector paper: [X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)
- Kaggle dataset contributors
- TensorFlow and Librosa communities

## Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

⭐ Star this repository if you find it helpful!
