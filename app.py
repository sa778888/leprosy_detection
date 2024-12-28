import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('leprosy_detection_model.h5')

# Define a function for prediction
def predict_leprosy(uploaded_image):
    img = uploaded_image.resize((64, 64))  # Resize image to match model input size
    img_array = image.img_to_array(img)  # Convert image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    
    prediction = model.predict(img_array)[0][0]
    if prediction >= 0.5:
        return "Not Leprosy", prediction
    else:
        return "Leprosy", prediction

# Streamlit App
st.title("Leprosy Detection App")
st.write("Upload an image to detect if it shows signs of leprosy.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    st.write("Analyzing...")
    label, confidence = predict_leprosy(image_data)
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence:.2f}**")
