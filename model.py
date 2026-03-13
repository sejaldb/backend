# backend/model.py
from PIL import Image

classifier = None

def load_model():
    global classifier
    if classifier is None:
        from transformers import pipeline
        MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
        classifier = pipeline("image-classification", model=MODEL_NAME, device=-1)
    return classifier

def predict(image: Image.Image):
    from PIL import Image
    image = image.convert("RGB")
    clf = load_model()
    try:
        results = clf(image)
    except Exception as e:
        return {"result": "ERROR", "confidence": 0.0, "error": str(e)}
    if not results:
        return {"result": "UNKNOWN", "confidence": 0.0}
    top = results[0]
    return {"result": top["label"], "confidence": float(top["score"])}