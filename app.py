import streamlit as st
import joblib
import cv2
import numpy as np
import os
import gdown
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

st.title("🩺 Pneumonia Detection System")

# =========================

# تحميل الموديلات من Google Drive

# =========================

def download_file(file_id, output):
if not os.path.exists(output):
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, output, quiet=False)

# 🔴 حط IDs هنا

download_file("15aVpQIskb6O3flwp2SreYeNDrHTDwr1U", "ensemble_model.pkl")
download_file("1r3e2WNZAXAmGEfKm8pRvEOXbzMWkcNQR", "pca_tool.pkl")
download_file("1RHgIS0cEYPq6q1BhOrQmHJPQQuvHZkEQ", "scaler_tool.pkl")

# =========================

# تحميل الموديلات

# =========================

final_model = joblib.load("ensemble_model.pkl")
final_pca = joblib.load("pca_tool.pkl")
final_sc = joblib.load("scaler_tool.pkl")

# تحميل MobileNetV2

model_net = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# =========================

# استخراج الخصائص

# =========================

def extract_features(img_rgb):
img_224 = cv2.resize(img_rgb, (224, 224))

```
f_deep = tf.keras.layers.GlobalAveragePooling2D()(
    model_net(preprocess_input(np.expand_dims(img_224, 0)))
).numpy().flatten()

gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

f_hog = hog(cv2.resize(gray_img, (64, 128)))

lbp = local_binary_pattern(cv2.resize(gray_img, (100, 100)), 8, 1)
hist, _ = np.histogram(lbp.ravel(), bins=10)

matrix = graycomatrix(gray_img, [5], [0], 256)
f_glcm = [graycoprops(matrix, 'dissimilarity')[0, 0]]

return np.concatenate([f_deep, f_hog, hist, f_glcm])
```

# =========================

# واجهة المستخدم

# =========================

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg","png"])

if uploaded_file:
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

```
st.image(img, caption="Uploaded Image")

features = extract_features(img)
features = final_pca.transform(final_sc.transform(features.reshape(1, -1)))

prob = final_model.predict_proba(features)[0][1]

st.write(f"Prediction Probability: {prob:.2f}")

if prob > 0.7:
    st.error("🚨 Pneumonia Detected")
else:
    st.success("✅ Normal")
```
