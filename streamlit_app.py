import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model (same folder me hona chahiye)
model = load_model("final_model.h5")

classes = ["flower", "not_flower"]

st.title("🌸 Flower vs Not Flower Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224,224))
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)

    pred_class = classes[np.argmax(pred)]
    confidence = np.max(pred)

    st.success(f"Prediction: {pred_class}")
    st.info(f"Confidence: {confidence*100:.2f}%")
