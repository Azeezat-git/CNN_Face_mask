import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split

# Define the Streamlit app layout
st.title("Face Mask Detection")

st.write("### Upload an image for prediction:")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Process image
    image = image.resize((128, 128))
    image = image.convert("RGB")
    image = np.array(image) / 255.0
    image = np.reshape(image, (1, 128, 128, 3))

    # Load the model
    model = keras.models.load_model("mask_detector_model.h5")

    # Make prediction
    prediction = model.predict(image)
    pred_label = np.argmax(prediction)

    if pred_label == 1:
        st.write("The person in the image is wearing a mask.")
    else:
        st.write("The person in the image is not wearing a mask.")


# Function to train the model (can be called manually if needed)
def train_model():
    st.write("### Training the model...")

    # Paths to data
    with_mask_path = "data/with_mask/"
    without_mask_path = "data/without_mask/"

    # Load and preprocess data
    file_names_masked = os.listdir(with_mask_path)
    file_names_no_mask = os.listdir(without_mask_path)

    data = []
    labels = []

    for img_file in file_names_masked:
        image = Image.open(with_mask_path + img_file)
        image = image.resize((128, 128))
        image = image.convert("RGB")
        image = np.array(image)
        data.append(image)
        labels.append(1)

    for img_file in file_names_no_mask:
        image = Image.open(without_mask_path + img_file)
        image = image.resize((128, 128))
        image = image.convert("RGB")
        image = np.array(image)
        data.append(image)
        labels.append(0)

    data = np.array(data)
    labels = np.array(labels)

    X_train, X_test, Y_train, Y_test = train_test_split(
        data, labels, test_size=0.2, random_state=2
    )
    X_train_scaled = X_train / 255.0
    X_test_scaled = X_test / 255.0

    # Build and compile model
    model = Sequential(
        [
            Conv2D(
                32, kernel_size=(3, 3), activation="relu", input_shape=(128, 128, 3)
            ),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(2, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"]
    )
    history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=5)

    # Save the model
    model.save("mask_detector_model.h5")
    st.write("Model trained and saved as 'mask_detector_model.h5'.")


# Button to start training the model
if st.button("Train Model"):
    train_model()
