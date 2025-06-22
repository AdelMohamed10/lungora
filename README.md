Lungora: Chest X-Ray Classifier training & API
A powerful and accessible deep learning API for chest X-ray image classification using PyTorch and Flask. This project identifies four classes: Covid, Pneumonia, Normal, and Undetected. It combines a PyTorch deep learning model with an OpenCV-HOG + Scikit-learn pre-classifier to ensure input validity.

Dataset
We used a merged and preprocessed dataset collected from several public sources, including:

(https://universe.roboflow.com/rellis-3d/covid-preumonia)[Roboflow]

Collected data for non-chest images

Preprocessing steps:

Resized all images to 224x224 (for ViT model) and 128x128 (for HOG+SVM).

Converted to grayscale for HOG features.

Applied data balancing and augmentation during training.

Model Training
The training pipeline includes:

Model: ViTForImageClassification (Vision Transformer from HuggingFace)

Loss: CrossEntropyLoss

Optimizer: AdamW

Epochs: 25

Frameworks: PyTorch + HuggingFace Transformers

Saved model format: state_dict for saving weights.

Training scripts are located in the trining notebook

API Structure
The API is built using:

Flask for web server

Flask-CORS for cross-origin support

gdown to fetch models from Google Drive

Sklearn for chest image validation (binary classifier)

TorchVision for image transformations

Endpoints:
GET /: Welcome message and model status.

POST /predict: Takes a chest X-ray image and returns:

predicted_label: (covid, pneumonia, normal, undetect)

confidence_score: model certainty

check_status: indicates whether the model is unsure

is_upload: indicates if the image is valid chest X-ray

