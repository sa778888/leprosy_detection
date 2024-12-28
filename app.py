import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained CNN model
cnn = load_model("./model.pkl")

# Define the Streamlit app
st.title("Binary Image Classification")
st.write("This app classifies images as 'yes' or 'no'.")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the uploaded image
    img = image.load_img(uploaded_file, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class
    result = cnn.predict(img_array)
    
    if result[0][0] == 1:
        prediction = "no"
    else:
        prediction = "yes"
    
    # Display the result
    st.write(f"Prediction: {prediction}")
