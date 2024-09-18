import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

# Load the pre-trained model
model_path = "mask_detector_model.h5"
model = keras.models.load_model(model_path)

# Define the Streamlit app layout
st.title("Face Mask Detection")

st.write("### Upload an image for prediction:")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and process the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    image = image.resize((128, 128))
    image = image.convert("RGB")
    image = np.array(image) / 255.0
    image = np.reshape(image, (1, 128, 128, 3))

    # Make prediction
    prediction = model.predict(image)
    pred_label = np.argmax(prediction)

    if pred_label == 1:
        st.write("The person in the image is wearing a mask.")
    else:
        st.write("The person in the image is not wearing a mask.")
