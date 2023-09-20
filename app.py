import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image, ImageDraw, ImageFont

# Load the Keras model
model = load_model("keras_model.h5", compile=False)

# Define class labels
class_labels = ["Aloevera", "Amla", "Amruta_Balli", "Arali", "Ashoka", "Ashwagandha", "Avacado", "Bamboo", "Basale", "Betel",
    "Betel_Nut", "Brahmi", "Castor", "Curry_Leaf", "Doddapatre", "Ekka", "Ganike", "Gauva", "Geranium", "Henna",
    "Hibiscus", "Honge", "Insulin", "Jasmine", "Lemon", "Lemon_grass", "Mango", "Mint", "Nagadali", "Neem",
    "Nithyapushpa", "Nooni", "Pappaya", "Pepper", "Pomegranate", "Raktachandini", "Class 37", "Sapota", "Tulasi",
    "Wood_sorel"]

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = Image.fromarray(img)
        img = img.resize((224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= (127.5) - 1
        prediction = model.predict(img_tensor)
        class_index = np.argmax(prediction[0])
        class_label = class_labels[class_index]

        # Create an image draw object and set font properties
        draw = ImageDraw.Draw(img)
        font_size = 20
        font = ImageFont.truetype("arial.ttf", font_size)
        font_color = (255, 0, 0)
        position = (10, 10)
        
        # Draw the class label on the image
        draw.text(position, f"Class: {class_label}", font=font, fill=font_color)

        return np.array(img)

def main():
    st.title("Object Detection with Webcam")

    # Create a button to start the webcam
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
