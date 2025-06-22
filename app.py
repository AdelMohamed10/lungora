import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
import gdown
import joblib
import traceback
import logging
from torchvision import transforms
import numpy as np
import cv2
from skimage.feature import hog

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Constants
model_path = "4classes.pth"
sk_model_path = "img_classifier.pkl"
file_id = "15jLV7id21txc_sBZ_rJfsFJZEwB3QYiM" ## Preparing another model if main model not found/exist
model_url = f"https://drive.google.com/uc?id={file_id}"
MAX_IMAGE_SIZE_MB = 10 # Max image size

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals
model = None
sk_model = None

# Class labels
class_names = ["covid", "normal", "pneumonia", "undetected"]

# Image transform for PyTorch model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Image resize and HOG params for SVM model
image_size = (128, 128)
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}


# Load sklearn model
def load_sklearn_model():
    global sk_model
    try:
        sk_model = joblib.load(sk_model_path)
        logger.info("Sklearn image classifier loaded.")
        return True
    except Exception as e:
        logger.error(f"Failed to load sklearn model: {str(e)}")
        traceback.print_exc()
        return False


# Fix missing dropout layers in ViT model
def fix_vit_dropout(model):
    """fix ViT dropout problem"""
    fixed_count = 0

    for name, module in model.named_modules():
        if hasattr(module, '__class__') and 'ViTSelfAttention' in str(module.__class__):
            if not hasattr(module, 'dropout'):
                # Getting dropout rate
                dropout_rate = 0.1  # default value
                if hasattr(model, 'config') and hasattr(model.config, 'attention_probs_dropout_prob'):
                    dropout_rate = model.config.attention_probs_dropout_prob
                elif hasattr(model, 'config') and hasattr(model.config, 'hidden_dropout_prob'):
                    dropout_rate = model.config.hidden_dropout_prob

                # Adding missing dropout layers
                module.dropout = torch.nn.Dropout(dropout_rate)
                fixed_count += 1
                logger.info(f"✅ Added dropout ({dropout_rate}) to {name}")

    if fixed_count > 0:
        logger.info(f"Fixed {fixed_count} missing dropout layers in ViT model")
    else:
        logger.info("No missing dropout layers found")

    return model


# Load PyTorch model
def load_model():
    global model
    try:
        if not os.path.exists(model_path):
            logger.info("Downloading PyTorch model...")
            gdown.download(model_url, model_path, quiet=False)

        # Load model with weights_only=False
        model = torch.load(model_path, map_location=device, weights_only=False)

        # Fix missing dropout layers
        model = fix_vit_dropout(model)

        # Move to device and set evaluation mode
        model.to(device)
        model.eval()

        # Double check that all modules are in eval mode
        for module in model.modules():
            if hasattr(module, 'training'):
                module.training = False

        logger.info("PyTorch model loaded and fixed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading PyTorch model: {str(e)}")
        traceback.print_exc()
        return False


# Preprocess image for sklearn using OpenCV and HOG
def preprocess_image_for_sklearn(image_bytes):
    try:
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image format")

        img = cv2.resize(img, image_size)
        features = hog(img, **hog_params)
        return features
    except Exception as e:
        logger.error(f"Error in sklearn preprocessing: {str(e)}")
        raise ValueError("Invalid image for sklearn model.")


# Preprocess image for PyTorch model
def preprocess_image_for_pytorch(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        return img_tensor
    except Exception as e:
        logger.error(f"Error in PyTorch preprocessing: {str(e)}")
        raise ValueError("Invalid image for PyTorch model.")


@app.route("/", methods=["GET"])
def home():
    status = "available" if model is not None else "Not Available"
    return jsonify({
        "message": "Welcome to X-Ray chest classifier!",
        "status": status,
        "classes": class_names
    })


@app.route("/predict", methods=["POST"])
def predict():
    is_success = False # API work or not
    is_upload = False # save image or not
    check_status = False # threshold-based message (handled on other side)
    certain = False # confidence > 95% or not (model made sure)

    if model is None or sk_model is None:
        if not load_model() or not load_sklearn_model():
            return jsonify({"is_success": False, "message": "Models not ready."}), 503

    try:
        if "image" not in request.files:
            return jsonify({"is_success": is_success, "message": "No image uploaded."}), 400

        file = request.files["image"]
        image_bytes = file.read()

        # Check image size
        file_size_mb = len(image_bytes) / (1024 * 1024)
        if file_size_mb > MAX_IMAGE_SIZE_MB:
            return jsonify({"is_success": False, "message": "Image exceeds size limit (10MB)."}), 413

        # Step 1: Use sklearn model to classify as chest or non-chest
        features = preprocess_image_for_sklearn(image_bytes).reshape(1, -1)
        prediction = sk_model.predict(features)[0]  # 1 = chest, 0 = non-chest

        if prediction == 0:
            return jsonify({
                "is_success": True,
                "is_upload": False
            }), 200

        # Step 2: PyTorch model for diagnosis
        inputs = preprocess_image_for_pytorch(image_bytes)

        # Ensure model is in evaluation mode
        model.eval()

        with torch.no_grad():
            outputs = model(inputs)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            elif isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        predicted_label = class_names[predicted_class.item()]
        confidence_score = confidence.item()

        if confidence_score <= 0.60:  ## Model cannot identify
            check_status = False
        elif confidence_score <= 0.95:  ## Model not sure enough
            check_status = True
        else:
            certain = True  ## Model sure of that

        is_success = True
        return jsonify({
            "is_success": is_success,
            "predicted_label": predicted_label,
            "is_upload": True,
            "check_status": check_status,
            "certain": certain
        })
    # Error handling
    except ValueError as ve:
        logger.error(f"Value error: {str(ve)}")
        return jsonify({"is_success": False, "message": "Invalid image."}), 400
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        traceback.print_exc()
        return jsonify({"is_success": False, "message": "Unexpected error."}), 500


# Load models on startup
load_model()
load_sklearn_model()

if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=5001)