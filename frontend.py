import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

st.title("Skin Cancer Classification using CNN")

# Load the saved model
model = load_model(r'C:\Users\PMLS\archive\my_model.h5')

# Read data
skin_df = pd.read_csv(r'C:\Users\PMLS\archive\HAM10000_metadata.csv')
SIZE = 32

# Define classes
label_encoder = LabelEncoder()
skin_df['label'] = label_encoder.fit_transform(skin_df['dx'])
classes = label_encoder.classes_

# Image classification
st.subheader("Image Classification")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='IMAGE UPLOADED', use_column_width=True)

    # Preprocess the image
    image = image.resize((SIZE, SIZE))
    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0  # Normalize the image

    # Make predictions
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    st.title(f"Predicted Class: {predicted_class}")