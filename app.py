import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import gdown
from torchvision import transforms
import traceback
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Model download details
file_id = "15jLV7id21txc_sBZ_rJfsFJZEwB3QYiM"
model_url = f"https://drive.google.com/uc?id={file_id}"
model_path = "XRay_Classifier.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Class labels
class_names = ["Covid", "Normal", "Pneumonia"]

descriptions = {
    "Covid": "🦠 فيروس كورونا (COVID-19) هو مرض فيروسي يصيب الجهاز التنفسي. الأعراض الشائعة تشمل الحمى، السعال الجاف، وضيق التنفس. يُنصح بالعزل المنزلي واستشارة الطبيب عند الحاجة.",
    "Normal": "✅ لا توجد علامات تدل على التهاب رئوي أو عدوى. يُنصح بالحفاظ على نمط حياة صحي، وشرب السوائل، واستشارة الطبيب عند الشعور بأي أعراض غير طبيعية.",
    "Pneumonia": "🫁 الالتهاب الرئوي هو عدوى تصيب الرئتين، وقد تسبب السعال مع البلغم، الحمى، وألمًا في الصدر. يُفضل مراجعة الطبيب للحصول على التشخيص الدقيق والعلاج المناسب."
}


def load_model():
    """Load the model from local storage or Google Drive."""
    global model
    try:
        if not os.path.exists(model_path):
            logger.info("⏳ Downloading model from Google Drive...")
            gdown.download(model_url, model_path, quiet=False)
            logger.info("✅ Model downloaded successfully!")

        logger.info(f"Loading model on {device}")
        model = torch.load(model_path, map_location=device)
        logger.info(f"✅ Loaded model: {model}")

        model.eval()
        logger.info("✅ Model ready for inference")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        traceback.print_exc()
        return False


def preprocess_image(image_bytes):
    """Preprocess image before making predictions."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        return img_tensor
    except Exception as e:
        logger.error(f"❌ Error processing image: {str(e)}")
        traceback.print_exc()
        raise ValueError("Failed to process input image")


@app.route("/", methods=["GET"])
def home():
    """API Home endpoint."""
    status = "متاح" if model is not None else "غير متاح (لم يتم تحميل النموذج)"
    return jsonify({
        "message": "مرحبًا بك في واجهة تصنيف صور الأشعة السينية!",
        "status": status,
        "classes": class_names
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Predict the class of the uploaded X-ray image."""
    if model is None:
        if not load_model():
            return jsonify({"error": "النموذج غير متاح حاليًا"}), 503

    try:
        if "image" not in request.files:
            return jsonify({"error": "لم يتم رفع أي صورة"}), 400

        image_bytes = request.files["image"].read()
        if not image_bytes:
            return jsonify({"error": "الصورة المرفوعة فارغة"}), 400

        inputs = preprocess_image(image_bytes)

        with torch.no_grad():
            outputs = model(inputs)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            elif isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            logger.info(f"🔍 Output shape: {outputs.shape}")
            logger.info(f"🔍 Output content: {outputs}")

            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        confidence_score = confidence.item()
        predicted_label = class_names[predicted_class.item()]
        all_probabilities = {class_names[i]: float(probabilities[0][i]) for i in range(len(class_names))}

        response = {
            "prediction": predicted_label,
            "confidence": f"{confidence_score:.2f}",
            "all_probabilities": all_probabilities
        }

        if confidence_score < 0.7:
            response["message"] = "⚠️ غير متأكد! يُرجى الرجوع إلى الطبيب."
        else:
            response["info"] = descriptions[predicted_label]

        return jsonify(response)

    except ValueError as ve:
        logger.error(f"Input error: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": "حدث خطأ داخلي"}), 500


# Load the model on startup
try:
    load_model()
except Exception as e:
    logger.error(f"Failed to load model on startup: {str(e)}")
    traceback.print_exc()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8081))
    app.run(port=port, debug=False, use_reloader=False)