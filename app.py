from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 🔹 Load model from Google Drive path
model = load_model("/content/drive/MyDrive/final_model.h5")

classes = ["flower", "not_flower"]

# 🔹 Take image path input
img_path = input("Enter image path: ")

# 🔹 Load and preprocess image
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# 🔹 Predict
pred = model.predict(img_array)

# 🔹 Output
pred_class = classes[np.argmax(pred)]
confidence = np.max(pred)

print("Prediction:", pred_class)
print("Confidence:", confidence)