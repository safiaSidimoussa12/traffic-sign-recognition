# 🚦 Traffic Sign Recognition

A web application that classifies German traffic signs using deep learning.

## Model
- Architecture: EfficientNetB0 (Transfer Learning)
- Dataset: GTSRB — German Traffic Sign Recognition Benchmark
- Classes: 43 traffic sign types
- Validation Accuracy: 98.6%
- Test Accuracy: 86.0%

## Tech Stack
- Python / Flask
- TensorFlow / Keras
- HTML / CSS

## Project Structure

traffic-sign-app/
├── app.py
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── uploads/
├── model/          ← place your .keras file here
├── requirements.txt
└── README.md

## Setup

```bash
# Create virtual environment
python -m venv traffic-sign-env

# Activate (Windows)
traffic-sign-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Note
The model file (`efficientnet_finetuned_final.keras`) is not included in this repository due to file size. Download it from the Kaggle notebook and place it in the `model/` folder.

## Results

| Model | Val Accuracy | Test Accuracy |
|---|---|---|
| Baseline CNN | ~51% | ~51% |
| ResNet50 (transfer learning) | 99.6% | 88.9% |
| EfficientNetB0 (transfer learning) | 98.6% | 86.0% |