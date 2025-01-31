from keras.api.models import load_model
import streamlit as st
from keras.api.preprocessing.image import load_img,img_to_array
import numpy as np
import gdown

# Load the trained model
gdown.download("https://drive.google.com/file/d/15IDZkEz12_AIjz6U6IMZ20LOCpRNwCKA/view?usp=sharing", output, quiet=False)

# Load the model
model = tf.keras.models.load_model(output)
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

# Streamlit interface
st.title("AbdulKader Javed Qureshi Potato Disease Detection")
st.write("Upload an image of a potato leaf to detect the disease.")

# Image upload
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = load_img(uploaded_file, target_size=(128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image)
    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)

    # Get the class labels from the train_generator

    # Get the predicted class label
    predicted_class_label = class_names[predicted_class_index]
    st.write(f'{predicted_class_label}')

    # Print the prediction
    print(f"Predicted class: {predicted_class_label}")
