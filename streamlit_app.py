import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('new_model.h5')

model = load_model()

# Define your image preprocessing function
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize the image to match the model's input shape
    image = np.asarray(image) / 255.0  # Normalize pixel values
    return image

# Define the Streamlit app
def main():
    st.title("Cat-Dog Image Classifier")
    st.write("Upload an image of DOG or CAT and let the model classify it!")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image of dog or cat...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Preprocess the image
            processed_image = preprocess_image(image)

           

            # Make predictions
            predictions = model.predict(np.expand_dims(processed_image, axis=0))


        

            

            # Display prediction results
            st.write("### Prediction Results:")
            if predictions[0][0] > 0.5:
                st.write("## It's a Dog!")
            else:
                st.write("## It's a Cat!")

        except Exception as e:
            st.write("Error processing image:", e)

# Run the app
if __name__ == "__main__":
    main()
