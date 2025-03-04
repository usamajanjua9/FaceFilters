import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import os

# Set up Streamlit UI
st.set_page_config(page_title="Webcam Face Detection & Filters", page_icon="📷", layout="wide")
st.title("📷 Webcam Face Detection, Filters, Emojis & Glasses Overlay 🎭")

# Load Haarcascade for Face Detection
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Sidebar Options
st.sidebar.header("🔧 Options")
emoji_option = st.sidebar.selectbox("😃 Choose an Emoji to Overlay:", 
                                    ("None", "😂", "😎", "😍", "🤩", "👽", "🐱"))

# Image Upload Option
st.sidebar.subheader("📷 Upload an Image")
uploaded_image = st.sidebar.file_uploader("📂 Choose an image", type=["jpg", "jpeg", "png"])

# Function to download and use an emoji-compatible font
def get_emoji_font():
    font_path = "NotoColorEmoji.ttf"

    if not os.path.exists(font_path):
        try:
            url = "https://github.com/googlefonts/noto-emoji/blob/main/fonts/NotoColorEmoji.ttf?raw=true"
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                with open(font_path, "wb") as f:
                    f.write(response.content)
                st.success("✅ Downloaded emoji font successfully!")
            else:
                st.warning("⚠ Failed to download emoji font.")
                return None
        except Exception as e:
            st.warning(f"⚠ Error downloading font: {e}")
            return None

    return font_path

# Function to overlay emoji using PIL
def overlay_emoji(frame, emoji_option):
    if emoji_option == "None":
        return frame

    # Convert OpenCV image to PIL format
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_frame)

    # Get emoji-compatible font
    font_path = get_emoji_font()
    if font_path:
        font = ImageFont.truetype(font_path, 60)  # Adjust size
    else:
        st.warning("⚠ Could not find an emoji-compatible font. Emojis may not render properly.")
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        draw.text((x + int(w / 3), y + int(h / 2)), emoji_option, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

# Process Uploaded or Captured Image
if uploaded_image:
    image = Image.open(uploaded_image)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame = overlay_emoji(frame, emoji_option)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, channels="RGB", caption="Processed Image 🎭")
    st.success("📸 Image processed successfully!")

st.sidebar.info("🚀 Developed by Dr. Usama Arshad")
