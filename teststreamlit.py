import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the Keras model
model = load_model("./keras_model.h5", compile=False)

# Define class labels
class_labels = ["Aloevera", "Amla", "Amruta_Balli", "Arali", "Ashoka", "Ashwagandha", "Avacado", "Bamboo", "Basale", "Betel",
    "Betel_Nut", "Brahmi", "Castor", "Curry_Leaf", "Doddapatre", "Ekka", "Ganike", "Gauva", "Geranium", "Henna",
    "Hibiscus", "Honge", "Insulin", "Jasmine", "Lemon", "Lemon_grass", "Mango", "Mint", "Nagadali", "Neem",
    "Nithyapushpa", "Nooni", "Pappaya", "Pepper", "Pomegranate", "Raktachandini", "Class 37", "Sapota", "Tulasi",
    "Wood_sorel"]

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= (127.5) - 1
        prediction = model.predict(img_tensor)
        class_index = np.argmax(prediction[0])
        class_label = class_labels[class_index]

        # Resize the camera frame for smoother display
        img = cv2.resize(img, (640, 480))

        # Adjust the font size and position to display class name
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2
        font_color = (0, 0, 255)
        x, y = 10, 40
        text_size = cv2.getTextSize(class_label, font, font_scale, font_thickness)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        if x + text_width > 640:
            x = 640 - text_width - 10

        img = cv2.putText(img, f"Class: {class_label}", (x, y), font, font_scale, font_color, font_thickness)

        return img

def main():
    st.title("Object Detection with Webcam")

    # Create a button to start the webcam
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
