import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import os

# Set up Streamlit UI
st.set_page_config(page_title="Webcam Face Detection & Filters", page_icon="📷", layout="wide")
st.title("📷 Webcam Face Detection, Filters, Emojis & Glasses Overlay 🎭")

# Load Haarcascade for Face Detection
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Sidebar Options
st.sidebar.header("🔧 Options")
filter_option = st.sidebar.radio("🎨 Choose a Filter:", 
                                 ("No Filter", "Grayscale", "Cartoon", "Blur", 
                                  "Edge Detection", "Pencil Sketch", "Sepia", 
                                  "Invert Colors", "Emboss", "Sharpen", "HSV"))

filter_intensity = st.sidebar.slider("🎛 Filter Intensity", 1, 10, 5)
brightness_factor = st.sidebar.slider("💡 Brightness", -100, 100, 0)

emoji_option = st.sidebar.selectbox("😃 Choose an Emoji to Overlay:", 
                                    ("None", "😂", "😎", "😍", "🤩", "👽", "🐱"))

st.sidebar.subheader("📷 Upload an Image")
uploaded_image = st.sidebar.file_uploader("📂 Choose an image", type=["jpg", "jpeg", "png"])

st.sidebar.subheader("📸 Take a Picture")
image_file = st.camera_input("Capture an Image")


# Function to fetch emoji images dynamically
def get_emoji_image(emoji_name):
    emoji_urls = {
        "😂": "https://emojicdn.elk.sh/😂",
        "😎": "https://emojicdn.elk.sh/😎",
        "😍": "https://emojicdn.elk.sh/😍",
        "🤩": "https://emojicdn.elk.sh/🤩",
        "👽": "https://emojicdn.elk.sh/👽",
        "🐱": "https://emojicdn.elk.sh/🐱"
    }
    
    if emoji_name not in emoji_urls:
        return None

    try:
        response = requests.get(emoji_urls[emoji_name], stream=True)
        if response.status_code == 200:
            emoji_img = Image.open(response.raw).convert("RGBA")
            return emoji_img
    except Exception as e:
        st.warning(f"⚠ Error loading emoji: {e}")
    
    return None


# Function to apply filters
def apply_filter(frame, filter_option, intensity):
    if filter_option == "Grayscale":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    elif filter_option == "Cartoon":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, intensity)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, intensity * 2, 250, 250)
        frame = cv2.bitwise_and(color, color, mask=edges)

    elif filter_option == "Blur":
        frame = cv2.GaussianBlur(frame, (intensity * 2 + 1, intensity * 2 + 1), 0)

    elif filter_option == "Edge Detection":
        frame = cv2.Canny(frame, intensity * 10, intensity * 20)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    elif filter_option == "Pencil Sketch":
        try:
            _, sketch = cv2.pencilSketch(frame, sigma_s=intensity * 10, sigma_r=0.07, shade_factor=0.05)
            frame = sketch
        except AttributeError:
            st.warning("⚠ Pencil Sketch filter is not available in your OpenCV version.")

    elif filter_option == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131], 
                           [0.349, 0.686, 0.168], 
                           [0.393, 0.769, 0.189]])
        frame = cv2.transform(frame, kernel * (intensity / 10))

    elif filter_option == "Invert Colors":
        frame = cv2.bitwise_not(frame)

    elif filter_option == "Emboss":
        kernel = np.array([[0, -intensity, -intensity], 
                           [intensity, 0, -intensity], 
                           [intensity, intensity, 0]])
        frame = cv2.filter2D(frame, -1, kernel)

    elif filter_option == "Sharpen":
        kernel = np.array([[0, -intensity, 0], 
                           [-intensity, intensity * 5, -intensity], 
                           [0, -intensity, 0]])
        frame = cv2.filter2D(frame, -1, kernel)

    elif filter_option == "HSV":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    return frame


# Function to overlay emoji image on detected faces
def overlay_emoji(frame, emoji_option):
    if emoji_option == "None":
        return frame

    emoji_img = get_emoji_image(emoji_option)
    if emoji_img is None:
        st.warning("⚠ Could not load emoji image.")
        return frame

    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        emoji_resized = emoji_img.resize((w, h // 3), Image.ANTIALIAS)
        pil_frame.paste(emoji_resized, (x, y), emoji_resized)

    return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)


# Process Uploaded or Captured Image
if uploaded_image or image_file:
    image = Image.open(uploaded_image) if uploaded_image else Image.open(image_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame = apply_filter(frame, filter_option, filter_intensity)
    frame = overlay_emoji(frame, emoji_option)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, channels="RGB", caption="Processed Image 🎭")
    st.success("📸 Image processed successfully!")

st.sidebar.info("🚀 Developed by Dr. Usama Arshad")
